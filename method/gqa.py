import torch
import torch.nn as nn

dropout_layer  =nn.Dropout(p=0.5)
# Dropout：随机丢弃，扩大其他部分元素，保持整体期望不变
t1 = torch.Tensor([1,2,3])
t2 = dropout_layer(t1)

print(t2)

# Linear:线性变换，对张量乘w再加b
layer = nn.Linear(in_features=3,out_features=5,bias=True)
t1 = torch.Tensor([1,2,3])
t2 = torch.Tensor([[1,2,3]]) #shape:(1,3)
#这里应用的w和b是随机的
output2 = layer(t2) #shape:(1,5)
print(output2)

# view函数：改变张量/矩阵 形状，元素数量不变
t = torch.tensor([[1,2,3,4,5,6],[7,8,9,10,11,12]]) #shape:(2,6)
t_view1 = t.view(3,4)
print(t_view1)  #shape:(3,4)
t_view2 = t.view(4,3)
print(t_view2)  #shape:(4,3)

# transpose:交换维度 / 矩阵转置
t = torch.tensor([[1,2,3],[4,5,6]]) #shape:(2,3)
t = t.transpose(0,1)    #交换第0维和第1维
print(t)

# triu：对角线下置零 ，diagonal表示向右上一条线偏移+0（用来掩码计算
x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(torch.triu(x,diagonal=0))
print(torch.triu(x,diagonal=1))

# reshape（类似于view
x = torch.arange(1,7)   #[1,2,3,4,5,6]
y = torch.reshape(x,(2,3))
print(y)
# 使用-1，自动判断另一个维度
z = torch.reshape(x,(3,-1))
print(z)