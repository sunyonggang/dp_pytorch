# 动手深度学习第二课



## 过拟合、欠拟合及解决方案
1. 过拟合、欠拟合
2. 权重衰减
3. 丢弃法

#### 过拟合、欠拟合
1. 过拟合是训练误差小但是测试误差大，欠拟合是训练误差和测试误差都大
2. 常见解决过拟合的方法：1. 加数据，2. 简化模型，3. 加正则


#### 权重衰减
1. L2范数正则化：添加惩罚项，使学出来的模型参数比较少


#### 丢弃发
1. 以一定的概率丢弃单元，同时拉升剩下的
2. 代码
    
    ```
    def dropout(X, drop_prob):
        X = X.float()
        assert 0 <= drop_prob <= 1
        keep_prob = 1 - drop_prob
        # 这种情况下把全部元素都丢弃
        if keep_prob == 0:
            return torch.zeros_like(X)
        mask = (torch.rand(X.shape) < keep_prob).float()
        
        return mask * X / keep_prob
    ```
    
## 梯度爆炸和消失
1. 当模型层数比较多的时候，模型的数值稳定性会变差
2. 深度模型有关数值稳定性的典型问题是消失（vanishing）和爆炸（explosion）


## 卷积神经网络基础
1. 卷积层
2. 池化层
3. 填充、步伐
4. 输入输出通道

#### 卷积层
1. 二维互相关（cross-correlation）运算的输入是一个二维输入数组和一个二维核（kernel）数组，输出也是一个二维数组，其中核数组通常称为卷积核或过滤器（filter）

#### 池化层
1. 对一个卷积核做max/avg之类的

#### 填充、步伐
1. 填充：原卷积向外填充X
2. 步伐：卷积核每次移动的步伐


## LeNet
1. LeNet的基本结构
    
    ```
    net = torch.nn.Sequential(     #Lelet                                                  
        Reshape(),
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), #b*1*28*28  =>b*6*28*28
        nn.Sigmoid(),                                                       
        nn.AvgPool2d(kernel_size=2, stride=2),                              #b*6*28*28  =>b*6*14*14
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),           #b*6*14*14  =>b*16*10*10
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),                              #b*16*10*10  => b*16*5*5
        Flatten(),                                                          #b*16*5*5   => b*400
        nn.Linear(in_features=16*5*5, out_features=120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )
    ```

## 深度卷积神经网络_part2
1. AlexNet
2. VGG
3. GoogLeNet


## 机器翻译及相关技术
1. 概念：将一段文本从一种语言自动翻译为另一种语言，用神经网络解决这个问题通常称为神经机器翻译（NMT）。 主要特征：输出是单词序列而不是单个单词
2. Seq2Seq

## 注意力机制
1. Attention 是一种通用的带权池化方法，输入由两部分构成：询问（query）和键值对（key-value pairs）