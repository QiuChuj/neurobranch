import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse
import math
from collections import namedtuple
from torch.nn.parameter import Parameter
import time
from torch.nn import MSELoss
import pickle

class MLP(nn.Module):
    def __init__(self, params, d_in, d_outs, name=None, nl_at_end=False):
        """
        参数:
        params: 配置字典，包含'mlp_transfer_fn'和'weight_reparam'等设置
        d_in: 输入维度
        d_outs: 各层输出维度的列表
        name: 模块名称（可选）
        nl_at_end: 是否在最后一层应用非线性激活
        """
        super(MLP, self).__init__()
        self.params = params
        self.name = name
        self.nl_at_end = nl_at_end
        self.transfer_fn = self._decode_transfer_fn(params['mlp_transfer_fn'])
        self.weight_reparam = params.get('weight_reparam', False)
        
        # 初始化权重和偏置
        self.ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        self.gs = nn.ParameterList() if self.weight_reparam else None
        
        d_current = d_in
        for i, d_out in enumerate(d_outs):
            # 初始化权重
            w = nn.Parameter(torch.Tensor(d_current, d_out))
            nn.init.xavier_uniform_(w)  # Xavier初始化
            self.ws.append(w)
            
            # 初始化偏置
            b = nn.Parameter(torch.zeros(d_out))
            self.bs.append(b)
            
            # 权重重参数化处理
            if self.weight_reparam:
                g = nn.Parameter(torch.ones(1, d_out))
                self.gs.append(g)
                
            d_current = d_out

    def _decode_transfer_fn(self, tf_name):
        """将TF激活函数名映射到PyTorch函数"""
        mapper = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': lambda x: F.leaky_relu(x, 0.2),
            'elu': F.elu,
            'selu': F.selu
        }
        return mapper.get(tf_name.lower(), F.relu)  # 默认使用ReLU

    def forward(self, z):
        """前向传播"""
        x = z
        for i in range(len(self.ws)):
            w = self.ws[i]
            
            # 权重重参数化处理
            if self.weight_reparam:
                w_normalized = F.normalize(w, p=2, dim=0)  # L2归一化
                g = self.gs[i]
                w = w_normalized * g  # 应用缩放因子
            
            # 线性变换
            x = torch.matmul(x, w) + self.bs[i]
            
            # 应用非线性激活（检查是否最后一层）
            if self.nl_at_end or i < len(self.ws) - 1:
                x = self.transfer_fn(x)
                
        return x
    

class NeuroBranch(nn.Module):
    def __init__(self, params):
        super(NeuroBranch, self).__init__()
        self.params = params
        self.d = self.params['d']
        self.n_rounds = self.params['n_rounds']
        self.device = torch.cuda.get_device_name(0)
        
        # 初始化可学习参数
        self.L_init_scale = Parameter(torch.tensor(1.0 / math.sqrt(params['d'])))
        self.C_init_scale = Parameter(torch.tensor(1.0 / math.sqrt(params['d'])))
        self.LC_scale = Parameter(torch.tensor(params['LC_scale']))
        self.CL_scale = Parameter(torch.tensor(params['CL_scale']))
        
        # 创建多层更新模块
        self.L_updates = nn.ModuleList([
            MLP(params, 2 * params['d'] + params['d'], 
                repeat_end(params['d'], params['n_update_layers'], params['d']),
                name=f"L_u_{t}" if not params['repeat_layers'] else "L_u",
                nl_at_end=params['mlp_update_nl_at_end'])
            for t in range(params['n_rounds'])
        ])
        
        self.C_updates = nn.ModuleList([
            MLP(params, params['d'] + params['d'], 
                repeat_end(params['d'], params['n_update_layers'], params['d']),
                name=f"C_u_{t}" if not params['repeat_layers'] else "C_u",
                nl_at_end=params['mlp_update_nl_at_end'])
            for t in range(params['n_rounds'])
        ])
        
        # 变量评分模块
        self.V_score = MLP(params, 2 * params['d'], 
                           repeat_end(params['d'], params['n_score_layers'], 1),
                           name="V_score", nl_at_end=False)

    def forward(self, vars, clauses, CL_indexes, position_indexes):
        # 解包参数
        n_vars = vars
        n_lits = 2 * n_vars
        n_clauses = clauses
        CL_idxs = CL_indexes
        pos_idxs = position_indexes
        
        # 构建稀疏矩阵 CL (clause-literals) 和转置 LC (literal-clauses)
        #! 对应的是文章中的G和G^T
        # print("位置矩阵的形状：")
        # print(pos_idxs.shape[1])
        values = torch.ones(pos_idxs.shape[1], device=CL_idxs.device)
        CL_sparse = torch.sparse_coo_tensor(
            pos_idxs, values, 
            size=(n_clauses, n_lits)
        )
        LC_sparse = torch.sparse_coo_tensor(
            pos_idxs.flip([0]), values,
            size=(n_lits, n_clauses)
        ).coalesce()

        # 初始化文字和子句状态
        L = torch.ones(n_lits, self.d, device=CL_idxs.device) * self.L_init_scale
        C = torch.ones(n_clauses, self.d, device=CL_idxs.device) * self.C_init_scale
        
        # 定义文字翻转函数
        def flip(lits):
            return torch.cat([lits[n_vars:], lits[:n_vars]], dim=0)
        
        # 消息传递循环
        for t in range(self.n_rounds):
            C_old, L_old = C, L
            
            # 文字到子句的消息传递
            LC_msgs = torch.sparse.mm(CL_sparse, L) * self.LC_scale
            C = self.C_updates[t](torch.cat([C, LC_msgs], dim=-1))
            C = self._normalize(C, self.params['norm_axis'], self.params['norm_eps'])
            if self.params['res_layers']:
                C = C + C_old
            
            # 子句到文字的消息传递
            CL_msgs = torch.sparse.mm(LC_sparse, C) * self.CL_scale
            L = self.L_updates[t](torch.cat([L, CL_msgs, flip(L)], dim=-1))
            L = self._normalize(L, self.params['norm_axis'], self.params['norm_eps'])
            if self.params['res_layers']:
                L = L + L_old
        
        # 生成变量评分
        V = torch.cat([L[:n_vars], L[n_vars:]], dim=-1)
        V_scores = self.V_score(V).squeeze(-1)
        # print("变量评分的数据类型：")
        # print(V_scores.dtype)
        
        # 返回命名元组
        NeuroSATGuesses = namedtuple('NeuroSATGuesses', ['pi_core_var_logits'])
        return NeuroSATGuesses(pi_core_var_logits=V_scores)
    
    def _normalize(self, x, axis, eps):
        """张量归一化"""
        if axis == 0:
            return (x - x.mean(0)) / (x.std(0) + eps)
        else:  # axis = 1
            return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + eps)
        
    def save(self, path):
        """保存模型参数到指定路径"""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """从指定路径加载模型参数"""
        self.load_state_dict(torch.load(path))
        
    def train_epoch(self, train_loader, val_loader, optimizer, device, 
                      epochs=50, save_path='/home/richard/project/neurobranch/models/EasySAT/neurobranch_model.pth'):
        """
        训练NeuroBranch模型的完整流程
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器实例
            device: 训练设备 (cpu/cuda)
            epochs: 训练轮数
            save_path: 模型保存路径
            
        返回:
            train_losses: 各轮训练损失
            val_losses: 各轮验证损失
        """
        self.train()  # 设置模型为训练模式
        criterion = MSELoss()  # 均方误差损失函数
        train_losses = []
        val_losses = []
        
        # 将模型移到指定设备
        self.to(device)
        
        for epoch in range(epochs):
            # 训练阶段
            epoch_train_loss = 0.0
            start_time = time.time()
            
            for args_batch, labels_batch in train_loader:
                # 解包批次数据
                vars = torch.tensor(args_batch[0]).to(device)
                clauses = torch.tensor(args_batch[1]).to(device)
                CL_batch = torch.tensor(args_batch[2][0], dtype=torch.int32).to(device)
                pos_batch = torch.tensor(args_batch[3], dtype=torch.int32).to(device)
                labels_batch = torch.tensor(labels_batch[0], dtype=torch.float32).to(device)
                
                # 前向传播
                optimizer.zero_grad()
                output = self.forward(vars, clauses, CL_batch, pos_batch)
                
                # 计算损失
                loss = criterion(output.pi_core_var_logits, labels_batch)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # 计算平均训练损失
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            # model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for args_batch, labels_batch in val_loader:
                    # 解包批次数据
                    vars = torch.tensor(args_batch[0]).to(device)
                    clauses = torch.tensor(args_batch[1]).to(device)
                    CL_batch = torch.tensor(args_batch[2][0], dtype=torch.int32).to(device)
                    pos_batch = torch.tensor(args_batch[3], dtype=torch.int32).to(device)
                    labels_batch = torch.tensor(labels_batch[0], dtype=torch.float32).to(device)
                    
                    output = self.forward(vars, clauses, CL_batch, pos_batch)
                    loss = criterion(output.pi_core_var_logits, labels_batch)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # 打印训练信息
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs} | '
                f'Train Loss: {avg_train_loss:.6f} | '
                f'Val Loss: {avg_val_loss:.6f} | '
                f'Time: {epoch_time:.2f}s')
        
        # 保存训练好的模型
        self.save(self.state_dict(), save_path)
        print(f'Model saved to {save_path}')
        
        return train_losses, val_losses
    
    def apply(self, data_loader, device, model_path='/home/richard/project/neurobranch/models/EasySAT/neurobranch_model.pth'):
        """
        应用预训练的NeuroBranch模型进行推理
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器实例
            device: 训练设备 (cpu/cuda)
            model_path: 预训练模型路径
            
        返回:
            output: 模型输出
        """

        # 加载模型并移到指定设备
        self.load(model_path)
        self.eval()
        self.to(device)
        
        for args_batch, labels_batch in data_loader:
            # 解包批次数据
            vars = torch.tensor(args_batch[0]).to(device)
            clauses = torch.tensor(args_batch[1]).to(device)
            CL_batch = torch.tensor(args_batch[2][0], dtype=torch.int32).to(device)
            pos_batch = torch.tensor(args_batch[3], dtype=torch.int32).to(device)
            
            # 前向传播
            output = self.forward(vars, clauses, CL_batch, pos_batch)
        
        return output.pi_core_var_logits
        
        

# 辅助函数
def repeat_end(val, n, k):
    return [val for _ in range(n)] + [k]