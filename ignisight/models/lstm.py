import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.io import savemat


num_time_steps = 8  # 训练时时间窗的步长
input_size = 7  # 输入数据维度
hidden_size = 32  # 隐含层维度
output_size = 1  # 输出维度
num_layers = 1  # 隐含层层数
num_heads = input_size  # 多头注意力层的头数


# 定义神经网络模型
class Net1(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(Net1, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # 当 batch_first设置为True时，输入的参数顺序变为：x：[batch, seq_len, input_size]，h0：[batch, num_layers, hidden_size]
            # 什么是batch：批量大小，就是一次传入的序列（句子）的数量。
            # 什么是seq：序列长度，即单词数量。
            # 什么是feature：特征长度，每个单词向量（Embedding）的长度。
            batch_first=True,  # 批次维度是第一个维度
        )
        for p in self.lstm.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):  # hidden_prev为RNN隐藏状态
        out, hidden_prev = self.lstm(x, hidden_prev)
        # [b, seq, h]
        out = out.view(-1, hidden_size)
        out = self.linear(
            out
        )  # [seq,h] => [seq,3]  seq=num_time_steps-1,1用来判断预测的结果
        out = out.unsqueeze(dim=0)  # => [1,seq,3]
        return out, hidden_prev