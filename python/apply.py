# from loaddata import create_data_loader
# from neurobranch import NeuroBranch
# import torch
# import json
# import pandas as pd

# # 模型配置参数
# with open("/home/richard/project/neurobranch/configs/apply/params.json", 'r') as f:
#     params = json.load(f)

# # 训练配置
# with open("/home/richard/project/neurobranch/configs/apply/config.json", 'r') as f:
#     config = json.load(f)

# config['params'] = params

# data_loader = create_data_loader(
#         clause_dir=config['clause_dir'],
#         score_dir=config['score_dir'],
#         batch_size=config['batch_size'],
#         max_clauses=config['max_clauses'],
#         max_vars=config['max_vars']
#     )

# net = NeuroBranch(config['params'])
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# output = net.apply(data_loader, device)

# output = output.detach().cpu().numpy()
# output = pd.DataFrame(output)
# output.to_csv('/home/richard/project/neurobranch/results/output.csv', mode='w',index=False, header=False)


import sysv_ipc
import ctypes
import numpy as np
import time
from neurobranch import NeuroBranch
import json
from loaddata import create_data_loader


# 定义与C结构体完全匹配的ctypes结构
class SharedData(ctypes.Structure):
    _fields_ = [
        ("features", ctypes.c_int * 20000 * 2),
        ("result", ctypes.c_double * 500),
        ("ready", ctypes.c_int),
        ("n_vars", ctypes.c_int),
        ("n_clauses", ctypes.c_int),
    ]


def tensor_to_c_array(tensor, target_size=500):
    """
    高效地将Tensor转换为c_double数组
    """
    # 转换为numpy并确保数据类型
    np_array = tensor.detach().cpu().numpy().astype(np.float64).flatten()

    # 处理大小不匹配
    if len(np_array) < target_size:
        np_array = np.pad(
            np_array,
            (0, target_size - len(np_array)),
            mode="constant",
            constant_values=0,
        )
    elif len(np_array) > target_size:
        np_array = np_array[:target_size]

    # 直接通过内存复制（高效方法）
    c_array = (ctypes.c_double * target_size)()
    ctypes.memmove(
        ctypes.addressof(c_array),
        np_array.ctypes.data,
        target_size * ctypes.sizeof(ctypes.c_double),
    )

    return c_array


def python_nn_process():
    try:
        # 模型配置参数
        with open(
            "/home/richard/project/neurobranch/configs/apply/params.json", "r"
        ) as f:
            params = json.load(f)
        # 训练配置
        with open(
            "/home/richard/project/neurobranch/configs/apply/config.json", "r"
        ) as f:
            config = json.load(f)
        config["params"] = params
        net = NeuroBranch(config["params"])

        # 生成相同的key（与C程序一致）
        key = sysv_ipc.ftok("/tmp/neurobranch", 83)

        # 计算结构体大小
        struct_size = ctypes.sizeof(SharedData)

        try:
            shm_existing = sysv_ipc.SharedMemory(key)
            shm_existing.remove()
        except sysv_ipc.ExistentialError:
            pass

        # 创建/连接共享内存
        shm = sysv_ipc.SharedMemory(key, sysv_ipc.IPC_CREAT, size=struct_size)
        sem = sysv_ipc.Semaphore(key, sysv_ipc.IPC_CREAT)

        # print(f"Python端连接成功，共享内存大小: {struct_size} 字节")

        while True:
            sem.acquire()  # 加锁

            # 读取共享内存数据
            shared_memory_data = shm.read()

            # 将字节数据转换为结构体
            shared_struct = SharedData.from_buffer_copy(shared_memory_data)

            # 检查数据是否就绪
            if shared_struct.ready == 1:
                # print("检测到C端数据就绪!")

                # 提取特征数组（转换为numpy数组便于处理）
                n_vars = shared_struct.n_vars
                n_clauses = shared_struct.n_clauses
                features_array = np.array(shared_struct.features)
                # print(f"特征向量形状: {features_array.shape}")
                data_loader = create_data_loader(n_vars, n_clauses, features_array)
                # print("dataloader创建完毕")

                # 模拟神经网络处理
                # 这里可以调用你的神经网络模型
                print("Apply Neurobranch ··· ···")
                nn_result = net.apply(data_loader)
                # print("apply end")

                # 更新结果
                shared_struct.result = tensor_to_c_array(nn_result)
                shared_struct.ready = 2  # 标记Python处理完成

                # print(f"神经网络计算结果: {nn_result}")

                # 将更新后的结构体写回共享内存
                shm.write(ctypes.string_at(ctypes.byref(shared_struct), struct_size))

            sem.release()  # 解锁
            #! 这里会不会导致decision时间变长？
            time.sleep(0.1)  # 避免过度占用CPU

    except KeyboardInterrupt:
        print("\nPython程序正常退出")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    print("这里是neurobranch进程的起始输出。")
    python_nn_process()
