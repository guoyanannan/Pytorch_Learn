import torch
import torchvision

x = torch.tensor(2.0,requires_grad=True)
w = torch.tensor(1.0,requires_grad=True)

a = torch.add(x,w)
b = torch.add(w,1)
y = torch.mul(a,b)
a.retain_grad()
b.retain_grad()


#反向传播
y.backward()
#非叶子节点是不保留梯度值的  print(w.grad)->None
print(w.grad)


#查看叶子结点
print(f'叶子结点:{x.is_leaf}{w.is_leaf}{a.is_leaf} \
{b.is_leaf}{y.is_leaf}')
#查看梯度
print(f'梯度值:{x.grad},{w.grad},{a.grad},{b.grad},{y.grad}')
