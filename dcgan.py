# 导入必要的库（边写边导入）
import cv2
import os
import argparse
from parser import args
import numpy as np
import random
#网络搭建需要的库
import torch
from torch.autograd import Variable
import torch.nn as nn
#准备数据集所需的库
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image


# 判断cuda是否存在，是否使用cuda张量
# device = torch.device('cuda:0' if (torch.cuda.is_available) else 'cup')
cuda = True if torch.cuda.is_available else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor# 转换为cuda张量还是普通张量


# 数据集准备
dataset = datasets.ImageFolder(root=args.dataset_path,# 注意这里是文件夹路径并非文件路径
        transform=transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))]
        )
)


# 数据集加载器
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers
    )


# 搭建生成器
class generator(nn.Module):# Module，Sequential大写
    def __init__(self):
        super(generator,self).__init__()
        self.init_size = args.input_size // 16
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim,(args.gf*8)*(self.init_size**2)))
        self.main = nn.Sequential(
            nn.BatchNorm2d(num_features=args.gf*8),# 只优化4维空间的第2维
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=args.gf*8, out_channels=args.gf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=args.gf*4, out_channels=args.gf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=args.gf*2, out_channels=args.gf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=args.gf, out_channels=args.input_channels, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(num_features=args.input_channels),
            nn.Tanh()
        )
    def forward(self,z):
        out = self.l1(z)
        out = out.view(out.shape[0],args.gf*8,self.init_size,self.init_size)# n,c,h,w
        out = self.main(out)
        return out


# 搭建判别器
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()            
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.input_channels, args.df, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.df) x 32 x 32
            nn.Conv2d(args.df, args.df * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.df * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.df*2) x 16 x 16
            nn.Conv2d(args.df * 2, args.df * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.df * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.df*4) x 8 x 8
            nn.Conv2d(args.df * 4, args.df * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.df * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (args.df*8) x 4 x 4
            nn.Conv2d(args.df * 8, 1, 4, 1, 0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )
        self.l1 = nn.Sequential(
            nn.Linear(128*args.input_size*args.input_size,1),
            nn.Sigmoid()
        )
    def forward(self,input):
        out = self.main(input)
        out = out.view(out.shape[0],-1)
        return out


# 权重初始化
def weights_init_normal(m):# m一般指网络框架
    classname = m.__class__.__name__    # m作为一个形参，原则上可以传递很多的内容，为了实现多实参传递，每一个moudle要给出自己的name. 所以这句话就是返回m的名字。
    if classname.find('Conv') != -1:#find()函数，实现查找classname中是否含有Conv字符，没有返回-1；有返回0.
        nn.init.normal_(m.weight.data, 0.0, 0.02)#m.weight.data表示需要初始化的权重。 nn.init.normal_()表示随机初始化采用正态分布，均值为0，标准差为0.02.
    elif classname.find('BatchNorm') != -1:           
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0) # nn.init.constant_()表示将偏差定义为常量0 



# 实例化、权重初始化
generator=generator()
discriminator=discriminator()
if args.pre_train:
    generator.load_state_dict(torch.load(args.g_model_path))
    discriminator.load_state_dict(torch.load(args.d_model_path))
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
#print(generator,discriminator)


# 优化器
optimizer_g = torch.optim.Adam(generator.parameters(),args.g_lr,betas=(args.beta1,args.beta2))#beta1,2的设置参考斋藤康毅
optimizer_d = torch.optim.Adam(discriminator.parameters(),args.d_lr,betas=(args.beta1,args.beta2))


# 损失函数(交叉熵)
loss=nn.BCELoss()


# 训练使用GPU&cuda并行计算
if cuda:
    generator.cuda()
    discriminator.cuda()
    loss.cuda()


# 训练
if args.train:
    print(f'Starting Training Loop...')
    generator.train()
    discriminator.train()
    for epoch in range(args.n_epochs):
        for i,(imgs,_) in enumerate(data_loader):# 加括号解决列表不是矩阵的问题，从而解决没有shape的问题
            #生成标签
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            #处理输入图片
            real_imgs = Variable(imgs.type(Tensor),)
            z = Variable(Tensor(np.random.normal(0,1,(imgs.shape[0],args.latent_dim))))
            # 使用Variable必须转换成张量，和np的矩阵还不同.variable只对张量进行封装
            gent_imgs = generator(z)

            # 训练判别器
            for k in range(args.k_disc):
                discriminator.zero_grad()
                discriminator_real_imgs = discriminator(real_imgs)
                d_x = discriminator_real_imgs.mean().item()
                discriminator_gent_imgs = discriminator(gent_imgs.detach())
                d_g_z1 = discriminator_gent_imgs.mean().item()
                real_loss = loss(discriminator_real_imgs,valid)
                real_loss.backward()# 求梯度
                fake_loss = loss(discriminator_gent_imgs,fake)#必须是用gent_imgs.detach()来切断G中的反向传播,因为从后边优化器知，其实只更新了d的参数
                fake_loss.backward()
                d_loss = real_loss + fake_loss# 梯度相加
                #d_loss.backward()
                optimizer_d.step()# 更新
            
            # 训练生成器
            # 前期训练
            #if (epoch*len(data_loader)+i+1) <= args.early:
                
            generator.zero_grad()
            discriminator_gent_imgs = discriminator(gent_imgs)
            d_g_z2 = discriminator_gent_imgs.mean().item()
            g_loss = loss(discriminator_gent_imgs,valid)
            g_loss.backward()
            optimizer_g.step()
            '''
            # 后期训练
            else:
                if (epoch*len(data_loader)+i+1) == args.early:
                    print(f'转换生成器损失函数')
                generator.zero_grad()
                discriminator_gent_imgs = discriminator(gent_imgs)
                d_g_z2 = discriminator_gent_imgs.mean().item()
                g_loss = (-1)*loss(discriminator_gent_imgs,valid)# 有问题
                g_loss.backward()
                optimizer_g.step()
            '''

            print('[Epoch: %d/%d, Batch: %d/%d, g_loss: %.4f, d_loss: %.4f, D(x): %.4f, D(G(z)): %.4f/%.4f]'
                    %(epoch, args.n_epochs, i, len(data_loader), g_loss.item(), d_loss.item(), d_x, d_g_z1, d_g_z2)
            )
            batches_done = epoch*len(data_loader) + i
            if batches_done % 500 == 0:
                save_image(gent_imgs.data[:24],'images/%d.jpg'%batches_done,)
                torch.save(generator.state_dict(),args.models_path+'g_%d.pth'%batches_done)
                torch.save(discriminator.state_dict(),args.models_path+'d_%d.pth'%batches_done)

    print('finished!')


# 测试
if args.test:
    z = Tensor(np.random.normal(0,1,(args.n_test,args.latent_dim)))
    g_model = generator
    if cuda:
        g_model.cuda()
    g_model.load_state_dict(torch.load(args.g_model_path), strict=True)
    g_model.eval()
    # g_model = g_model.to(device)

    print('g_Model path {:s}. \nTesting...'.format(args.g_model_path))
    with torch.no_grad():
        output = g_model(z).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    leng = output.shape[0]
    for i in range(leng):
        output1 = output[i]
        output1 = np.transpose(output1[[2, 1, 0], :, :], (1, 2, 0))
        output1 = (output1 * 255.0).round()
        os.makedirs('./results', exist_ok=True)
        cv2.imwrite('results/%d_rlt.png'%i, output1)