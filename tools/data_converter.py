"""
对 data_gen 之后的 npy 数据做进一步处理
"""

import os
import numpy as np


# input_npy = r"..\data\standard_9.npy"
input_dir = r"..\data\testdata"

keypointsnpy = []
labelsnpy = []
failed_files = []

for file in os.listdir(input_dir):
    if not file.endswith('.npy'):
        print(f"跳过非NPY文件: {file}")
        continue
    filepath = os.path.join(input_dir, file)
    # arr = np.load(filepath, allow_pickle=True).item()
    try:
        # 加载NPY文件
        arr = np.load(filepath, allow_pickle=True).item()

        # 验证数据结构
        if "keypoints" not in arr or "label" not in arr:
            print(f"文件格式错误: {file} - 缺少必要的键")
            failed_files.append(file)
            continue
        # 提取关键点和标签
        keypointsnpy.append(arr["keypoints"])
        labelsnpy.append(int(arr["label"]))

        # print(f"成功处理: {file}")

    except Exception as e:
        print(f"处理文件失败: {file} - 错误: {str(e)}")
        failed_files.append(file)

    # keypointsnpy.append(arr["keypoints"])
    # labelsnpy.append(int(arr["label"]))

if keypointsnpy and labelsnpy:
    print(len(keypointsnpy))
    print(len(labelsnpy))
    np.save("../data/test_keypoints/test_keypoints.npy", np.asarray(keypointsnpy))
    np.save("../data/test_keypoints/test_labels.npy", np.asarray(labelsnpy))

    if failed_files:
        print(f"\n警告: {len(failed_files)} 个文件处理失败")
        print(f"失败文件列表: {failed_files}")
else:
    print("\n错误: 没有成功处理任何文件!")
    if failed_files:
        print(f"所有 {len(failed_files)} 个文件均处理失败")