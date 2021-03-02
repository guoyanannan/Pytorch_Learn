import torch
import torchvision

x = torch.tensor(2.0,requires_grad=True,device='cuda')
w = torch.tensor(1.0,requires_grad=True,device='cuda')

a = torch.add(x,w)
b = torch.add(w,1)
y = torch.mul(a,b)
#保留非叶子节点梯度，须在反向传播前定义
a.retain_grad()
b.retain_grad()


#反向传播 ->计算梯度
y.backward()
#非叶子节点是不保留梯度值的  print(w.grad)->None
print(w.grad)

#查看叶子结点
print(f'叶子结点:{x.is_leaf}{w.is_leaf}{a.is_leaf}{b.is_leaf}{y.is_leaf}')
#查看梯度
print(f'梯度值:{x.grad},{w.grad},{a.grad},{b.grad},{y.grad}')
#查看note创建时所用到的方法
print(f'method:{x.grad_fn},{w.grad_fn},{a.grad_fn},{b.grad_fn},{y.grad_fn}')  #type:tensor
#查看是否保留梯度(虽然是True但是非叶子节点会自动清空)
print(f'保留梯度:{x.requires_grad}{w.requires_grad}{a.requires_grad}{b.requires_grad}{y.requires_grad}')
