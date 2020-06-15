#PyTorch实现自动编码器（多层感知机、卷积神经网络） & 变分自编码器

#多层感知机AE
'''
编码器：4层网络，中间使用ReLU激活函数，最后输出维度是3维
解码器：输入3维，输出28×28的图像数据
解码器最后的Tanh()将最后的输出转换到-1 ～1之间，这是因为我们
输入的图片已经变换到了-１～1之间了，这里的输出必须和其对应
'''
class autoencoder1(nn.Module):
    def __init__(self):
        super(autoencoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,12),
            nn.ReLU(True),
            nn.Linear(12,3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(True),
            nn.Linear(12,64),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,28*28),
            nn.Tanh() 
        )
    def forward(self,x):
        x = encoder(x)
        x = decoder(x)
        return x
        
'''
将多层感知机换成卷积神经网络
解码器：转置卷积（通过计算已知输入，求未知输入的过程）
根据你要上采样的大小判断stride大小，然后根据stride导致
的像素分布情况确定你的kernel size的视野大小，最后保证
-stride+2padding+kernel_size=0即可
（W - 1）* stride + output_padding + kernel_size - 2* padding
'''
class autoencoder2(nn.Module):
    def __init__(self):
        super(autoencoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,stride=3,padding=1), #b 16,10,10
            nn.ReLU(True),
            nn.MaxPool2d(2,stride=2),   #b 16,5,5
            nn.Conv2d(16,8,3,stride=2,padding=1), # 8,3,3
            nn.ReLU(True),
            nn.MaxPool2d(2,stride=1) # 8,2,2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8,16,3,stride=2), # 16,5,5
            nn.ReLU(True),
            nn.ConvTranspose2d(16,8,5,stride=3,padding=1),#8,15,15
            nn.ReLU(True),
            nn.ConvTranspose2d(8,1,stride=2,padding=1),#1,28,28
            nn.Tanh()
        )
    def forward(self,x):
        x = self.encoder(x)
        x = self.encoder(x)
        return x









