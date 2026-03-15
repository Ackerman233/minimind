import torch

out = torch.zeros(5)
index = torch.tensor([0,1,3])
src = torch.tensor([1.,2.,3.])

out.scatter_add_(dim=0,index=index,src=src)
print(out)

out2 = torch.tensor(
    ([1.,2.,3.],
     [4.,5.,6.])
)
print(out2.mean(dim=1))

# moeFFN用的

x=torch.tensor([1,2,3])
print( torch.repeat_interleave(x,repeats=2) )


y=torch.tensor([3,1,2])
idx= torch.argsort(y)
print(idx)
print(y[idx])


z=torch.tensor([0,1,1,3,2,1])
count = torch.bincount(z)
print(count)