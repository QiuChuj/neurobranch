import torch
from torch.optim import Adam
from neurobranch import NeuroBranch
from loaddata import SATDataset, create_data_loaders
import json

# 完整训练流程
def full_training_pipeline(config):
    """
    NeuroBranch模型的完整训练流程
    
    参数:
        config: 包含所有配置的字典
    
    返回:
        训练好的模型
    """
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        clause_dir=config['clause_dir'],
        score_dir=config['score_dir'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        max_clauses=config['max_clauses'],
        max_vars=config['max_vars']
    )
    
    # 初始化模型
    net = NeuroBranch(config['params'])
    
    # 设置优化器
    optimizer = Adam(net.parameters(), lr=config['learning_rate'])
    
    # 设置训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 训练模型
    train_losses, val_losses = net.train(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=config['epochs'],
        save_path=config['save_path']
    )
    
    # 返回训练结果
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

# 示例配置和使用
if __name__ == "__main__":
    # 模型配置参数
    with open("/home/richard/project/neurobranch/configs/train/params.json", 'r') as f:
        params = json.load(f)
    
    # 训练配置
    with open("/home/richard/project/neurobranch/configs/train/config.json", 'r') as f:
        config = json.load(f)
    
    config['params'] = params
    
    # 执行完整训练流程
    training_results = full_training_pipeline(config)
    print("训练完成!")