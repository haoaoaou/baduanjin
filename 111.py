# # import torch
# # print( torch.cuda.is_available())   # 检查cuda是否可用
# # print(torch.version.cuda)          # 查看cuda版本
# #
# # print(torch.backends.cudnn.is_available())
# # print(torch.backends.cudnn.version() )
#
# # from pathlib import Path
# #
# # model_path = r"C:\Users\76897\.cache\rtmlib\hub\checkpoints\rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.onnx"
# # print(f"模型是否存在: {Path(model_path).exists()}")
#
# # import os
# # model_dir = os.path.expanduser("~/.cache/rtmlib/hub/checkpoints")
# # model_path = os.path.join(model_dir, "rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.onnx")
# # print(f"模型路径类型: {type(model_path)}")
# # print(f"模型路径值: {model_path}")
# # print(f"模型文件是否存在: {os.path.exists(model_path)}")
#
# import os
#
# data_path = r"data/testdata"
#
# # 检查目录是否存在
# if not os.path.exists(data_path):
#     print(f"错误: 目录 '{data_path}' 不存在")
# else:
#     # 检查是否为目录
#     if not os.path.isdir(data_path):
#         print(f"错误: '{data_path}' 不是一个目录")
#     else:
#         # 检查是否有读取权限
#         if not os.access(data_path, os.R_OK):
#             print(f"错误: 没有权限读取目录 '{data_path}'")
#         else:
#             print(f"目录 '{data_path}' 存在且可以访问")
#             # 列出目录内容
#             print(f"目录内容: {os.listdir(data_path)}")
#

import numpy as np
test = np.load(r"E:\PyCharmProject\PoseClassifier\data\testdata\动作9-8-49.npy", allow_pickle=True)
print(test)
