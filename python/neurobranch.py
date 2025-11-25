import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse
import math
from collections import namedtuple
from torch.nn.parameter import Parameter
import time
# import pandas as pd
from datetime import datetime
from torch.nn import MSELoss

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.new_model = params['new_model']
        self.train_mode = params['train_mode']
        self.model_path = params['model_path']
        self.forward_time = 0
        
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
        
        # 将模型移到指定设备
        if not self.train_mode:
            # 加载模型并移到指定设备
            self.load()
            self.eval()
        elif not self.new_model:
            self.load()
            self.train()
        else:
            self.train()
        self.to(self.device)

    def forward(self, vars, clauses, position_indexes):
        # 解包参数
        # print("初始化参数：")
        n_vars = vars
        n_lits = 2 * n_vars
        n_clauses = clauses
        pos_idxs = position_indexes
        
        # 构建稀疏矩阵 CL (clause-literals) 和转置 LC (literal-clauses)
        #! 对应的是文章中的G和G^T

        # print("构建稀疏矩阵：")
        values = torch.ones(pos_idxs.shape[1], device=self.device)
        # print(pos_idxs.shape, pos_idxs.shape[0], pos_idxs.shape[1])/
        # print(values.shape)
        #! 这句出现问题：number of dimensions must be sparse_dim (1) + dense_dim (0), but got 2
        CL_sparse = torch.sparse_coo_tensor(
            pos_idxs, values, 
            size=(n_clauses, n_lits),
        )
        # print("2")
        LC_sparse = torch.sparse_coo_tensor(
            pos_idxs.flip([0]), values,
            size=(n_lits, n_clauses),
        ).coalesce()

        # print("初始化文字和子句状态：")
        # print(n_lits)
        # print(n_clauses)
        # print(self.d)
        # print(self.L_init_scale.data)
        # print(self.C_init_scale.data)
        # print(self.LC_scale.data)
        # print(self.CL_scale.data)
        L = torch.ones([n_lits, self.d], device=self.device) * self.L_init_scale
        C = torch.ones([n_clauses, self.d], device=self.device) * self.C_init_scale
        # print(L.data)
        # print(C.data)
        # 定义文字翻转函数
        def flip(lits):
            return torch.cat([lits[n_vars:], lits[:n_vars]], dim=0)
        
        # 消息传递循环
        for t in range(self.n_rounds):
            # print("Round : ", t)
            C_old, L_old = C, L
            
            # 文字到子句的消息传递
            LC_msgs = (torch.sparse.mm(CL_sparse, L) * self.LC_scale)
            # print("LC_msgs:")
            # print(LC_msgs.data)
            C = self.C_updates[t](torch.cat([C, LC_msgs], dim=-1))
            C = self._normalize(C, self.params['norm_axis'], self.params['norm_eps'])
            if self.params['res_layers']:
                C = C + C_old
            # print("C:")
            # print(C.data)
            # 子句到文字的消息传递
            CL_msgs = (torch.sparse.mm(LC_sparse, C) * self.CL_scale)
            # print("CL_msgs:")
            # print(CL_msgs.data)
            L = self.L_updates[t](torch.cat([L, CL_msgs, flip(L)], dim=-1))
            L = self._normalize(L, self.params['norm_axis'], self.params['norm_eps'])
            if self.params['res_layers']:
                L = L + L_old
            # print("L:")
            # print(L.data)
        
        # 生成变量评分
        V = torch.cat([L[:n_vars], L[n_vars:]], dim=-1)
        V_scores = self.V_score(V).squeeze(-1)
        # print("V_scores:")
        # print(V_scores.shape)
        
        # 返回命名元组
        NeuroSATGuesses = namedtuple('NeuroSATGuesses', ['pi_core_var_logits'])
        return NeuroSATGuesses(pi_core_var_logits=V_scores)
    
    def _normalize(self, x, axis):
        """张量归一化"""
        if axis == 0:
            return (x - x.mean(0)) / x.std(0)
        else:  # axis = 1
            return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True)
        
    def min_max_normalize(self, x):
        """最小-最大归一化"""
        min_x = x.min()
        max_x = x.max()
        return (x - min_x) / (max_x - min_x + 1e-8)
    
    def save(self):
        """保存模型参数到指定路径"""
        torch.save(self.state_dict(), self.model_path)
    
    def load(self):
        """从指定路径加载模型参数"""
        self.load_state_dict(torch.load(self.model_path))
        
    def train_epoch(self, train_loader, val_loader, optimizer, 
                      epochs=1):
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
        criterion = MSELoss()  # 均方误差损失函数
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            epoch_train_loss = 0.0
            start_time = time.time()
            i = 0
            
            for args_batch, labels_batch in train_loader:
                i += 1
                print("Training: file ", i)
                # 解包批次数据
                # print("Constructing tensors:")
                vars = args_batch[0].detach().to(self.device)
                clauses = args_batch[1].detach().to(self.device)
                pos_batch = torch.tensor(args_batch[2], dtype=torch.int32).to(self.device)
                labels_batch = labels_batch.detach().to(self.device)
                
                # 前向传播
                # print("Forward:")
                optimizer.zero_grad()
                output = self.forward(vars, clauses, pos_batch)
                
                # 计算损失
                # print("Loss computation:")
                #! 这里添加simp版本的loss计算
                #! 将loss分为8个1*1000的部分计算，然后取平均
                label_chuncks = min_max_normalize_chunks(labels_batch)
                loss = 0.0
                output_chunk = adjust_tensor_simple(output.pi_core_var_logits)
                for j in range(8):
                    loss += criterion(output_chunk, label_chuncks[j])
                loss = loss / 8.0
                
                # loss = criterion(output.pi_core_var_logits, labels_batch)
                
                # 反向传播
                # print("Loss backward:")
                loss.backward()
                #! 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                # print("Loss backward complete.")
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
                    print("Evaluating:")
                    # 解包批次数据
                    vars = args_batch[0].detach().to(self.device)
                    clauses = args_batch[1].detach().to(self.device)
                    pos_batch = torch.tensor(args_batch[2], dtype=torch.int32).to(self.device)
                    labels_batch = labels_batch[0].detach().to(self.device)
                    
                    output = self.forward(vars, clauses, pos_batch)
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
        self.save()
        print(f'Model saved to {self.model_path}')
        
        return train_losses, val_losses
    
    def apply(self, data_loader):
        """
        应用预训练的NeuroBranch模型进行推理
        
        参数:
            data_loader: 训练数据加载器
            optimizer: 优化器实例
            device: 训练设备 (cpu/cuda)
            model_path: 预训练模型路径
            
        返回:
            output: 模型输出
        """
        
        #! 这里有问题，加载不出来
        # print("加载器加载 ··· ···")
        for args_batch in data_loader:
            # 解包批次数据
            # print(args_batch.shape)
            # print("数据解包 ··· ···")
            vars = args_batch[0].detach().to(self.device)
            clauses = args_batch[1].detach().to(self.device)
            pos_batch = torch.tensor(args_batch[2][0], dtype=torch.int32).to(self.device)

            # 前向传播
            # start = time.perf_counter()
            # print("前向传播 ··· ···")
            output = self.forward(vars, clauses, pos_batch)
            # end = time.perf_counter()
            # self.forward_time = end - start
            # self.log_time_to_csv()
        
        return output.pi_core_var_logits
    
    # def log_time_to_csv(self, filename='/home/richard/project/neurobranch/time.csv'):
    #     """
    #     使用 pandas 将计时结果（单次一个时间）写入或追加到 CSV 文件。

    #     Args:
    #         duration (float): 通过 time.perf_counter() 计算得到的时间间隔（秒）。
    #         filename (str, optional): CSV 文件名。默认为 'timings.csv'.
    #         label (str, optional): 本次计时的标签或描述。默认为 None.

    #     Returns:
    #         None
    #     """
    #     # 创建新数据行
    #     new_row = pd.DataFrame({'duration_seconds': [self.forward_time]})

    #     try:
    #         # 尝试读取现有文件
    #         existing_df = pd.read_csv(filename)
    #         # 如果文件存在，将新行追加到现有数据的末尾
    #         updated_df = pd.concat([existing_df, new_row], ignore_index=True)
    #     except FileNotFoundError:
    #         # 如果文件不存在，新数据就是初始数据
    #         updated_df = new_row

    #     # 将 DataFrame 写回 CSV，不保留索引列
    #     updated_df.to_csv(filename, index=False)
        
        

# 辅助函数
def repeat_end(val, n, k):
    return [val for _ in range(n)] + [k]

def print_memory_usage(stage):
    print(f"== {stage} ==")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB\n")
    
def min_max_normalize_chunks(tensor, epsilon=1e-8):
    """
    将8x1000张量拆分为8个1x1000张量并分别进行Min-Max归一化
    
    参数:
        tensor: 形状为[8, 1000]的输入张量
        epsilon: 防止除零的小常数，默认1e-8
    
    返回:
        normalized_chunks: 归一化后的张量列表，包含8个形状为[1, 1000]的张量
    """
    # 检查输入张量形状
    if tensor.dim() != 2:
        raise ValueError("输入张量必须是二维的")
    
    if tensor.shape[0] != 8 or tensor.shape[1] != 1000:
        print(f"警告: 输入张量形状为{tensor.shape}，期望形状为[8, 1000]")
    
    # 使用torch.chunk沿第0维拆分成8个1x1000的张量[1](@ref)
    chunks = torch.chunk(tensor, chunks=8, dim=0)
    
    normalized_chunks = []
    
    # 对每个块单独进行Min-Max归一化[2,5](@ref)
    for i, chunk in enumerate(chunks):
        # 计算当前块的最小值和最大值
        min_val = chunk.min()
        max_val = chunk.max()
        
        # 处理所有值相同的情况（避免除零）
        if max_val - min_val < epsilon:
            # 如果所有值相同，归一化到0
            normalized_chunk = torch.zeros_like(chunk)
        else:
            # 应用Min-Max归一化公式: (x - min) / (max - min)
            normalized_chunk = (chunk - min_val) / (max_val - min_val + epsilon)
        
        normalized_chunks.append(normalized_chunk)
    
    return normalized_chunks

def adjust_tensor_simple(tensor, target_shape=(1, 1000)):
    """
    使用PyTorch内置函数简化调整过程
    """
    # 展平为一维
    flattened = tensor.flatten()
    
    # 使用pad函数进行填充（更高效）
    if len(flattened) < target_shape[1]:
        padding = target_shape[1] - len(flattened)
        # 在末尾填充0
        padded = torch.nn.functional.pad(flattened, (0, padding))
    else:
        padded = flattened[:target_shape[1]]
    
    return padded.unsqueeze(0)  # 添加批次维度