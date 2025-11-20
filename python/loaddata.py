import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple

# 定义SAT数据集类
class SATDataset(Dataset):
    """
    用于加载和处理SAT问题数据的PyTorch数据集
    
    参数:
        clause_dir: 包含CNF文件的目录路径
        score_dir: 包含CSV标签文件的目录路径
        max_clauses: 最大子句数（用于填充）
        max_vars: 最大变量数（用于填充）
    """
    def __init__(self, clause_dir, score_dir, max_clauses, max_vars):
        self.clause_dir = clause_dir
        self.score_dir = score_dir
        self.max_clauses = max_clauses
        self.max_vars = max_vars
        self.file_pairs = self._get_file_pairs()
        
        # 定义NeuroSATArgs
        self.NeuroSATArgs = []
    
    def _get_file_pairs(self):
        """获取匹配的CNF和CSV文件对"""
        clause_files = [f for f in os.listdir(self.clause_dir) if f.endswith('.cnf')]
        score_files = [f for f in os.listdir(self.score_dir) if f.endswith('.csv')]
        
        file_pairs = []
        for clause_file in clause_files:
            base_name = os.path.splitext(clause_file)[0]
            score_file = f"{base_name}.csv"
            if score_file in score_files:
                file_pairs.append((os.path.join(self.clause_dir, clause_file),
                                  os.path.join(self.score_dir, score_file)))
        
        return file_pairs
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        clause_path, score_path = self.file_pairs[idx]
        
        # 解析CNF文件
        n_vars, n_clauses, position_indexes = self._parse_clauses(clause_path)
        
        # 创建NeuroSATArgs对象
        self.NeuroSATArgs = [n_vars, n_clauses, position_indexes]
        args = self.NeuroSATArgs
        
        # 解析CSV文件获取标签
        scores = self._parse_scores(score_path, n_vars)
        
        return args, torch.tensor(scores, dtype=torch.float32)
    
    def _parse_clauses(self, clause_path):
        """
        解析CNF文件，提取子句信息
        
        返回:
            n_vars: 变量数量
            n_clauses: 子句数量
            position_indexes: 子句-文字索引张量 (形状: [2, ?])
        """
        clauses = []
        n_vars = 0
        n_clauses = 0
        
        with open(clause_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('c') or not line:
                    continue  # 跳过注释行和空行
                
                if line.startswith('p'):
                    # 解析问题行: p cnf <variables> <clauses>
                    parts = line.split()
                    n_vars = int(parts[2])
                    n_clauses = int(parts[3])
                else:
                    # 解析子句行
                    literals = [int(x) for x in line.split() if x != '0']
                    if literals:
                        clauses.append(literals)
        
        # 创建子句-文字索引，以备后面创建稀疏矩阵使用
        position_indexes = [[],[]]
        
        for i, clause in enumerate(clauses[:self.max_clauses]):
            for lit in clause:
                # 文字索引: 正数表示原变量，负数表示取反
                # 映射到 [0, 2*n_vars-1] 范围
                idx = lit - 1 if lit > 0 else (-lit - 1 + n_vars)
                if idx < 2 * self.max_vars:  # 确保在有效范围内
                    position_indexes[0].append(i)  # 子句索引
                    position_indexes[1].append(idx)  # 文字索引
        
        return n_vars, n_clauses, position_indexes
    
    def _parse_scores(self, score_path, n_vars):
        """
        解析CSV文件，获取文字得分标签
        
        返回:
            scores: 变量得分标签 (形状: [n_vars])
        """
        scores = np.zeros(n_vars, dtype=np.float64)
        idx = 0
        with open(score_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                scores[idx] = row[0]
                idx += 1
        return scores

#! 定义共享内存版数据类
class ApplyDataset(Dataset):
    def __init__(self, n_vars, n_clauses, pos_idx):
        super().__init__()
        self.n_vars = n_vars
        self.n_clauses = n_clauses
        self.position_indexes = pos_idx
        
    def __getitem__(self, idx):
        return [self.n_vars, self.n_clauses, self.position_indexes]
    
    def __len__(self):
        return 1

# 创建训练数据加载器
def create_data_loaders(clause_dir, score_dir, batch_size=1, val_split=0.2, 
                        max_clauses=1000, max_vars=1000):
    """
    创建训练和验证数据加载器
    
    参数:
        clause_dir: 子句文件目录
        score_dir: 得分文件目录
        batch_size: 批次大小
        val_split: 验证集比例
        max_clauses: 最大子句数
        max_vars: 最大变量数
        
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 创建完整数据集
    full_dataset = SATDataset(clause_dir, score_dir, max_clauses, max_vars)
    
    # 划分训练集和验证集
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader

#! 创建应用数据加载器
def create_data_loader(n_vars, n_clauses, pos_idxs):

    dataset = ApplyDataset(n_vars, n_clauses, pos_idxs)
    
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2
    )
    
    return data_loader