#!/bin/bash

# 定义文件夹路径
clauses_dir="/home/richard/project/neurobranch/dimacs/train/clauses/BMS_k3_n100_m429_0"
scores_dir="/home/richard/project/neurobranch/dimacs/train/scores/BMS_k3_n100_m429_0"

# 检查文件夹是否存在
if [ ! -d "$clauses_dir" ]; then
    echo "错误：clauses文件夹不存在 - $clauses_dir" >&2
    exit 1
fi

if [ ! -d "$scores_dir" ]; then
    echo "错误：scores文件夹不存在 - $scores_dir" >&2
    exit 1
fi

# 声明关联数组存储已见过的文件对哈希值
declare -A seen_hashes

echo "开始扫描文件对..."
echo "Clauses目录: $clauses_dir"
echo "Scores目录: $scores_dir"
echo "======================================"

# 遍历clauses目录下的所有.cnf文件
find "$clauses_dir" -maxdepth 1 -type f -name "*.cnf" | while read cnf_file; do
    # 获取文件名（不含路径和扩展名）
    base_name=$(basename "$cnf_file" .cnf)
    
    # 对应的csv文件路径
    csv_file="$scores_dir/$base_name.csv"
    
    # 检查对应的csv文件是否存在
    if [ -f "$csv_file" ]; then
        # 计算文件的MD5哈希值
        hash_cnf=$(md5sum "$cnf_file" | cut -d' ' -f1)
        hash_csv=$(md5sum "$csv_file" | cut -d' ' -f1)
        
        # 组合哈希值作为该文件对的唯一标识
        combined_hash="${hash_cnf}-${hash_csv}"
        
        # 检查是否已经存在相同内容的文件对
        if [ -n "${seen_hashes[$combined_hash]}" ]; then
            echo "删除重复文件对: $base_name"
            echo "  - 删除: $cnf_file"
            echo "  - 删除: $csv_file"
            
            # 实际删除文件（如先测试，可注释掉以下两行）
            rm "$cnf_file"
            rm "$csv_file"
        else
            # 第一次见到这个哈希组合，记录并保留文件对
            seen_hashes["$combined_hash"]=1
            echo "保留文件对: $base_name (首次出现)"
        fi
    else
        echo "警告: $cnf_file 没有对应的CSV文件" >&2
    fi
done

echo "======================================"
echo "去重操作完成！"