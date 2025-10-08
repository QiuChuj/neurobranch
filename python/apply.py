from loaddata import create_data_loader
from neurobranch import NeuroBranch
import torch
import json
import pandas as pd

# 模型配置参数
with open("/home/richard/project/neurobranch/configs/apply/params.json", 'r') as f:
    params = json.load(f)

# 训练配置
with open("/home/richard/project/neurobranch/configs/apply/config.json", 'r') as f:
    config = json.load(f)

config['params'] = params

data_loader = create_data_loader(
        clause_dir=config['clause_dir'],
        score_dir=config['score_dir'],
        batch_size=config['batch_size'],
        max_clauses=config['max_clauses'],
        max_vars=config['max_vars']
    )

net = NeuroBranch(config['params'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output = net.apply(data_loader, device)

output = output.detach().cpu().numpy()
output = pd.DataFrame(output)
output.to_csv('/home/richard/project/neurobranch/results/output.csv', mode='w',index=False, header=False)
