import torch
import torch.nn as nn

class EfficientNet(nn.Module):
    
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.stage1 = nn.Conv2d(3, 32, 3, padding = 1, bias = False)
        self.stage2 = MBConv(32, 16, 1, 3, downsize = True, se = False)
        self.stage3 = nn.Sequential(
            MBConv(16, 24, 6, 3, downsize = True),
            MBConv(24, 24, 6, 3)
        )
        self.stage4 = nn.Sequential(
            MBConv(24, 40, 6, 5, downsize = True),
            MBConv(40, 40, 6, 5)
        )
        self.stage5 = nn.Sequential(
            MBConv(40, 80, 6, 3, downsize = True),
            MBConv(80, 80, 6, 3),
            MBConv(80, 80, 6, 3)
        )
        self.stage6 = nn.Sequential(
            MBConv(80, 112, 6, 5),
            MBConv(112, 112, 6, 5),
            MBConv(112, 112, 6, 5),
        )
        self.stage7 = nn.Sequential(
            MBConv(112, 192, 6, 5, downsize = True),
            MBConv(192, 192, 6, 5),
            MBConv(192, 192, 6, 5),
            MBConv(192, 192, 6, 5)
        )
        self.stage8 = MBConv(192, 320, 6, 3)
        self.stage9 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
            flatten(),
            nn.Linear(1280, 1000)
        )

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.stage1(x)
        # N x 32 x 224 x 224
        x = self.stage2(x)
        # N x 16 x 112 x 112
        x = self.stage3(x)
        # N x 24 x 56 x 56
        x = self.stage4(x)
        # N x 40 x 28 x 28
        x = self.stage5(x)
        # N x 80 x 14 x 14
        x = self.stage6(x)
        # N x 112 x 14 x 14
        x = self.stage7(x)
        # N x 192 x 7 x 7
        x = self.stage8(x)
        # N x 320 x 7 x 7
        x = self.stage9(x)
        # N x 1000

        return x

class flatten(nn.Module):

    def forward(self, x):
        return torch.flatten(x, 1)

class MBConv(nn.Module):

    def __init__(self, inc, outc, expand_ratio, kernel_size, downsize = False, se = True):
        super(MBConv, self).__init__()
        self.inc, self.outc = inc, outc
        self.downsize = downsize
        self.block1 = BasicBlock(inc, inc * expand_ratio, swish = True)
        self.dw_conv = nn.Conv2d(inc * expand_ratio, inc * expand_ratio, kernel_size, stride = 2 if downsize else 1, padding = kernel_size // 2)
        self.se = SE_Block(inc * expand_ratio, 4) if se else nn.Sequential()
        self.block2 = BasicBlock(inc * expand_ratio, outc)

    def forward(self, x):
        res = x
        x = self.block1(x)
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.block2(x)
        if self.inc == self.outc and not self.downsize:
            x += res

        return x    

class BasicBlock(nn.Module):

    def __init__(self, inc, outc, swish = False):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(inc, outc, 1, bias = False)
        self.bn = nn.BatchNorm2d(outc)
        self.swish = self._swish if swish else nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.swish(x)

        return x

    def _swish(self, x):
        return x * torch.sigmoid(x)

class SE_Block(nn.Module):

    def __init__(self, inc, reduction_ratio):
        super(SE_Block, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(inc, inc // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(inc // reduction_ratio, inc)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ratio = self.pool(x)
        ratio = torch.flatten(ratio, 1)
        ratio = self.fc1(ratio)
        ratio = self.relu(ratio)
        ratio = self.fc2(ratio)
        ratio = self.sigmoid(ratio)
        output = x * ratio[:, :, None, None]

        return output 

x = torch.rand(5, 3, 224, 224)
model = EfficientNet()
y = model(x)
print(y.shape)
