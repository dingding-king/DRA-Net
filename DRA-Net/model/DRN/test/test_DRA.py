# -*-coding:utf-8-*-
"""
Created on 2024.9.17
programing language:python
@author:夜剑听雨
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.dataset import MyDataset
from utils.SignalProcessing import batch_snr
from model.UnetModel import Unet


# 指定加载数据的batch_size
batch_size = 8
# 加载测试数据集，不乱序
test_path_x = "..\\data\\train_patch\\sam\\"
test_path_y = "..\\data\\train_patch\\ori\\"
test_dataset = MyDataset(test_path_x, test_path_y)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# 加载网络，图片单通道1，分类为1。
model = Unet()

# 检测是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
temp_sets = []   # # 用于记录测试集的SNR,去噪前和去噪后都要记录
# 加载模型
file_list = glob.glob(os.path.join(".\\save_dir\\", '*pth'))
snr_set1 = 0.0
for i in range(len(file_list)):
    state_dict = torch.load(file_list[i])
    model.load_state_dict(state_dict)
    model.to(device=device)  # 模型拷贝至GPU
    model.eval()  # 开启评估模式
    # 测试集测试网络，采用计算一个batch数据的信噪比(snr)作为评估指标
    snr_set2 = 0.0
    for batch_idx, (test_x, test_y) in enumerate(test_loader, 0):
        # 加载数据至GPU
        test_x = test_x.to(device=device, dtype=torch.float32)
        test_y = test_y.to(device=device, dtype=torch.float32)
        with torch.no_grad():  # 不需要做梯度更新，所以要关闭求梯度
            out = model(test_x)  # 使用网络参数，输出预测结果
            # 计算网络去噪后的数据和干净数据的信噪比(此处是计算了所有的数据，除以了batch_size求均值)
            if i < 1:  # 不用每次都计算
                SNR1 = batch_snr(test_x, test_y)  # 去噪前的信噪比
            SNR2 = batch_snr(out, test_y)  # 去噪后的信噪比
        if i < 1:  # 不用每次都计算
            snr_set1 += SNR1
        snr_set2 += SNR2
    # 累加计算本次模型的loss，最后还需要除以可以抽取多少个batch数，即最后的count值
    if i < 1:  # 不用每次都计算
        snr_set1 = snr_set1 / (batch_idx + 1)
    snr_set2 = snr_set2 / (batch_idx + 1)
    # 测试的loss保存至temp_sets中
    temp_sets.append(snr_set2)
    # 使用format格式化输出，保留小数点后四位
    print("epoch={}，去噪前的平均信噪比(SNR)：{:.4f} dB，去噪后的平均信噪比(SNR)：{:.4f} dB".format(i + 1, snr_set1, snr_set2))
# fmt参数，指定保存的文件格式。将loss_sets存为txt文件

np.savetxt('./result/snr_sets_train.txt', temp_sets, fmt='%.4f')
# 显示snr曲线
res_snr = np.loadtxt('./result/snr_sets_train.txt')
x = range(len(res_snr))
fig = plt.figure()
plt.plot(x, res_snr)
plt.legend(['denoise_snr'])
plt.xlabel('epoch')
plt.ylabel('SNR')
plt.savefig('./result/snr_plot_train.png', bbox_inches='tight')
plt.tight_layout()

plt.show()
