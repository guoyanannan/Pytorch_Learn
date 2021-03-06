import numpy as np
import torch
import torchvision


#例子1：torch.tensor()：直接导入的方式
flag = False
if flag:
    # 均匀分布
    a = np.random.rand(5,2)
    print(f'ndarray数据类型：{a.dtype}',a)

    b = torch.tensor(a)
    print(f'Tensor的数据类型：{b.dtype}',b)

#例子2：从numpy创建tensor,改变tensor或numpy的值对方也随之改变
flag = False
if flag:
    r = np.array([[1,2],[3,4],[5,6]])
    r_tr = torch.from_numpy(r)
    print('原始数据ndarray:',r)
    print('原始数据tensor:',r_tr)

    r[1][0]=4
    print('修改ndarray:',r)
    print('原始数据tensor:',r_tr)

    r_tr[1][0] = 5
    print('原始数据ndarray:', r)
    print('修改tensor:', r_tr)


#例子3：依据数值创建张量
#3.1 torch.zeros()
flag = False
if flag:
    o_t = torch.tensor([2,2,2])

    print('原始张量o_t类型:', o_t.dtype)
    print('原始张量o_t:',o_t)

    ##out:将输出结果赋予变量:必须是张量？——
    c_t = torch.zeros(3,3,out=o_t)
    print('经过out赋值后的0_t',o_t)
    print('依据数值创建的张量0_t',c_t)

#3.2 torch.zeros_like()
flag = False
if flag:
    oo_t = torch.rand((3,3))
    #zeross_like(参数需要也是tensor)
    o_t = torch.zeros_like(oo_t)
    print(oo_t,'\n',o_t)

#3.3 torch.full_like()
#总结out输出值的接受变量类型需要和输出者保持一致
flag =False
if flag:
    a = torch.tensor([1])
    o_t = torch.full((3,3),1.3,out=a)
    print(o_t,'\n',a)
    o_l_t = torch.full_like(o_t,fill_value=4)
    print(o_l_t)


#3.4 torch.arange() 创建等差一维张量,torch.linspace()常见均分一维张量，torch.logspace()创建对数均分的一维张量，torch.eye()创建单位对角矩阵
flag = False
if flag:
    o_t = torch.arange(1,10,2) #argument 3：stride
    print(o_t,'\n',o_t.dtype)
    o_t = torch.linspace(1,10,5)#argument 3：number
    print(o_t,'\n',o_t.dtype)
    o_t = torch.logspace(1,10,3)#argument 3：number
    print(o_t,'\n',o_t.dtype)
    o_t = torch.eye(3,6)  #n,m 设置一个或者都设置都可以，设置一个创建方阵
    print(o_t, '\n', o_t.dtype)


#例子4 依概率分布创建张量
#4.1 torch.normal（） 创建正态分布的张量
flag = False
if flag:
    #mean:张量 std：张量
    mean = torch.arange(1,5,dtype=torch.float)  # type:torch.tensor
    std = torch.arange(1,5,dtype=torch.float)
    print('mean:',mean,mean.dtype)
    print('std:',std)
    #当参数1和参数2都是tensor时，无法定义形状，输出和max(mean，std)形状一样
    o_t = torch.normal(mean,std)
    print(o_t, '\n', o_t.dtype)
    #mean:张量 std：张量
    #当mean和std是float时，可以定义形状
    o_t = torch.normal(0.0,1.0,(3,3))
    print(o_t, '\n', o_t.dtype)

#4.2 torch.randn() 创建标准正态分布的张量,torch.randn_like()
#torch.rand()0-1之间创建均匀分布 torch.randint()创建整数型[Low,Hight)分布
#多维
flag = False
if flag:
    o_t = torch.randn((3,3))
    print(o_t, '\n', o_t.dtype)
    o_t = torch.randn_like(o_t,) #layout：在内存中的存储方式
    print(o_t, '\n', o_t.dtype)

    o_t = torch.rand((5,5))
    print(o_t, '\n', o_t.dtype)
    o_t = torch.ones((2,3))
    #rand_like()传入变量类型需要是tensor
    o_t = torch.rand_like(o_t)
    print(o_t, '\n', o_t.dtype)

    o_t = torch.randint(1,5,(3,3))
    print(o_t, '\n', o_t.dtype)
    o_t = torch.ones((2,3))  #type:torch.float
    print(o_t, '\n', o_t.dtype)
    o_t = torch.randint_like(o_t,1,5)
    print(o_t, '\n', o_t.dtype)

#4.3 排列分布，伯努利分布 ： torch.randperm()  torch.bernoulli()
flag = False
if flag:
    o_t = torch.randperm(4) #从0-3随机排列，默认torch.int64
    print(o_t, '\n', o_t.dtype)
    pre = torch.rand(1) #0-1之间的均匀分布
    print(pre)
    BNL_T = torch.bernoulli(pre)  #以input为概率，生成伯努利分布（非0即1）
    print(BNL_T, '\n', o_t.dtype)