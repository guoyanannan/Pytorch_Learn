import torch

'''
torch.manual_seed(args.seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
if args.cuda:
    torch.cuda.manual_seed(args.seed)#为当前GPU设置随机种子；
#如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子
'''
torch.manual_seed(10)


# ====================================== retain_graph ==============================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    # print(w,x)
    # print(w.size(),x.size())

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    #如果需要反向传播需要保留计算图,否则不会计算梯度值
    y.backward(retain_graph=True)   #实际调用的是torch.autograd.backward()
    print(w.grad)
    # y.backward()

# ====================================== (gradient)grad_tensor各tensor权值 ==============================================
# ====================================== 多loss构建以及加权方法 ========================================
# flag = True
flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True) #torch.Size([1])
    x = torch.tensor([2.], requires_grad=True) #torch.Size([1])

    a = torch.add(w, x)     # retain_grad()
    b = torch.add(w, 1)

    y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)    dy0/dw = 2*1+3*1=5
    y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2

    loss = torch.cat([y0, y1], dim=0)       # [y0, y1] ->[5,2]
    grad_tensors = torch.tensor([1., 2.])

    loss.backward(gradient=grad_tensors)    # gradient 传入 torch.autograd.backward()中的grad_tensors

    print(w.grad)  #5*1+2*2=9


# ====================================== autograd.gard ==============================================
# flag = True
flag = False
if flag:

    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)     # y = x**2
    #参数1：需要求导的tensor，参数2：对谁求导，参数3：是否对返回值继续求导，参数4：是否保存计算图，参数5：是否计算多维梯度权重
    grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2 * 3 = 6
    print(grad_1)

    grad_2 = torch.autograd.grad(grad_1[0], x)              # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
    print(grad_2)


#对应autograd三种特殊属性
# ======================================叶子节点梯度不会自动清零==============================================
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True) #tensor.size([1])
    x = torch.tensor([2.], requires_grad=True)

    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)
        y.retain_grad()

        y.backward()
        print(w.grad)  #->对w求导
        print(y.grad)  #->对y求导
        print(y)       #->因为没有进行参数值更新,故该值一直不变

        w.grad.zero_() #叶子节点需手动清零，否则梯度值累加
        #y.grad.zero_()  #执行与否不影响该结点梯度自动清零
    print('-----------------------更新参数看看---------------------------')
    for i in range(4):
        a = torch.add(w, x)
        b = torch.add(w, 1)
        y = torch.mul(a, b)
        y.retain_grad()

        y.backward()
        #更新参数w
        print(f'更新前w梯度值:{w.grad}')
        print(f'更新前y值:{y,y.requires_grad}')
        print(f'更新后w值:{w.data}')
        w.data.sub_(0.001 * w.grad)
        # print(f'更新后w值:{w.grad}')  #->对w求导
        #print(f'更新后y值:{y}')       #->因为没有进行参数值更新,故该值一直不变
        print('---------------分割线-------------------------')

        w.grad.zero_()
        #y.grad.zero_()  #执行与否不影响该结点梯度自动清零


# ======================================依赖叶子节点的的非叶子节点默认requires_grad=true,但也会默认清空(也即不进行赋值) ==============================================
# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(a.requires_grad, b.requires_grad, y.requires_grad)
    print(a.grad, b.grad, y.grad)


# ======================================叶子结点不可执行in-place ==============================================
flag = True
# flag = False
if flag:
    #该种方式是直接在地址上更改数值
    a = torch.ones((1, ))
    print(id(a), a)
    a += torch.ones((1, ))
    print(id(a), a)
    #该种方式是直接开辟一个新的地址存储a值
    a = torch.ones((1,))
    print(id(a), a)
    a = a + torch.ones((1, ))
    print(id(a), a)


# flag = True
flag = False
if flag:

    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)   #type:torch.Tensor
    y = torch.mul(a, b)
    #单独执行b,会报错：梯度计算所需的一个变量已经被inplace操作修改
    b.add_(1)
    print(id(b), b)
    #单独执行w，会报错：对叶子节点进行了in—place操作
    w.add_(1)
    print(id(w),w)
    """
    autograd小贴士：
        叶子节点梯度不自动清零 
        依赖于叶子结点的结点，requires_grad默认为True     
        叶子结点不可执行in-place 
    """
    y.backward()