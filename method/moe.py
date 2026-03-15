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

