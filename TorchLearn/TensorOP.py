import torch


# 张量的操作运算
####---------------------拼接和切分--------------------------####
flag = False
if flag:
    t = torch.ones((5,2))
    t_cats = torch.cat((t,t),dim=1)
    print(t_cats,'\n',t_cats.shape)
    t_1_cats = torch.stack((t,t),dim=0) #创建一个新的维度并在此进行拼接
    print(t_1_cats, '\n', t_1_cats.shape)
    print('-----------------chunks-------------------')
    t_chunks = torch.chunk(t,3,dim=0)
    for idx,t1 in enumerate(t_chunks):
        print(f'tensor_{idx}:{t1},形状为{t1.shape}')
    print('-----------------split-------------------')
    t_splits = torch.split(t,[1,3,1],dim=0)
    for idx,t2 in enumerate(t_splits):
        print(f'tensor_{idx}:{t2},形状为{t2.shape}')

####---------------------按照索引从新拼接张量--------------------------####
flag = False
if flag:
    ####---------------------依照index索引重新拼接的张量--------------------------####
    t = torch.ones((5,5))
    t[1:3,:] = 2
    print(f'原始数据:\n{t}')
    idx = torch.tensor([1,2,0],dtype=torch.long)
    t_selects = torch.index_select(t,dim=1,index=idx) #依照index索引重新拼接的张量
    print(f'重组后的tensor:\n{t_selects}')
    for id,t in enumerate(t_selects):
        print(f'tensor_{id}:{t},形状是{t.shape}')

    ####---------------------依照mask重新拼接的张量--------------------------####
    t1 = torch.randint(0,9,(3,3))
    print(f'原始数据t1:\n{t1}')
    mask = t1.ge(6)                   #type:tensor
    print(f"mask:\n{mask}")
    t2 = torch.rand((3,3)) *10
    print(f'原始数据t2:\n{t2}')
    t_mask = torch.masked_select(t2,mask)
    print(f'返回的一维张量t_mask:\n{t_mask}')

####---------------------张量的转换--------------------------####
Isreshape = False
if Isreshape:
    t = torch.randint(1,11,(2,5))
    t_r = torch.reshape(t,(-1,5,2))
    print(f'原始数据:\n{t}{t.shape}\n转换后数据:\n{t_r}{t_r.shape}')
Istranspose = False
if Istranspose:
    t = torch.normal(0,1,size=(3,3,1))  #hwc -cwh=chw
    t_t = torch.transpose(t,dim0=2,dim1=0)
    t_t = torch.transpose(t_t,dim0=2,dim1=1)

    print(f'原始:\n{t}{t.shape}\n转置:\n{t_t}{t_t.shape}')
Issqueeze = False
if Issqueeze:
    t = torch.randn((1,4,4,1))
    t_s = torch.squeeze(t)
    t_1 = torch.squeeze(t,dim=1)
    t_3 = torch.squeeze(t,dim=3)
    print(t_s,t_s.shape,'\n',t_1,t_1.shape,'\n',t_3,t_3.shape)
Isunsqueeze = False
if Isunsqueeze:
    t = torch.randperm(9)
    print(t,t.shape)
    t = torch.reshape(t,(-1,3))
    print(t,t.shape)
    t = torch.unsqueeze(t,dim=0)
    print(t,t.shape)

####---------------------张量的数学运算--------------------------####
op = False
if op:
    t0 = torch.randint(1,10,(3,3))
    print(t0)
    t1 = torch.ones_like(t0)
    print(t1)
    t3 = torch.add(t0,t1,alpha=2)
    print(t3)
