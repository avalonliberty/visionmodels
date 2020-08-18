import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.block1 = BasicBlock(3, 64, 11, 4, 2)
        self.block2 = BasicBlock(64, 192, 5, 1, 2)
        self.block3 = BasicBlock(192, 384, 3, 1, 1)
        self.block4 = BasicBlock(384, 256, 3, 1, 1)
        self.block5 = BasicBlock(256, 256, 3, 1, 1)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        # n x 3 x 224 x 224
        x = self.block1(x)
        # n x 64 x 55 x 55
        x = self.maxpool1(x)
        # n x 64 x 27 x 27
        x = self.block2(x)
        # n x 192 x 27 x 27
        x = self.maxpool2(x)
        # n x 192 x 13 x 13
        x = self.block3(x)
        # n x 384 x 13 x 13
        x = self.block4(x)
        # n x 256 x 13 x 13
        x = self.block5(x)
        # n x 256 x 13 x 13
        x = self.avgpool(x)
        # n x 256 x 6 x 6
        x = torch.flatten(x, 1)
        # n x 9216
        x = self.classifier(x)

        return x

class BasicBlock(nn.Module):
    
    def __init__(self, inc, outc, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()
        self.kernel = nn.Conv2d(inc, outc, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.kernel(x)
        x = self.relu(x)

        return x