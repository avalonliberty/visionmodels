import torch
import torch.nn as nn

class InvertedResidual(nn.Module):

    def __init__(self, inc, outc, stride, groups):
        super(InvertedResidual, self).__init__()

        self.stride = stride
        self.hidden_features = outc // 4 
        if self.stride > 1:
            outc -= inc

        if self.stride > 1:
            self.branch1 = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1)
        else:
            self.branch1 = nn.Sequential()

        self.branch2_p1 = nn.Sequential(
            nn.Conv2d(inc, self.hidden_features, 1, groups = groups, bias = False),
            nn.BatchNorm2d(self.hidden_features),
            nn.ReLU(inplace = True)
        )
        self.branch2_p2 = nn.Sequential(
            nn.Conv2d(self.hidden_features, self.hidden_features, 3, self.stride, 1, groups = self.hidden_features, bias = False),
            nn.BatchNorm2d(self.hidden_features),
            nn.Conv2d(self.hidden_features, outc, 1,  groups = 2, bias = False),
            nn.BatchNorm2d(outc)
        )
    
    def forward(self, x):
        # Type: Tensor(N x Channel x Height x Width)
        route1 = self.branch1(x)
        route2 = self.branch2_p1(x)
        route2 = self.channel_shuffle(route2, 2)
        route2 = self.branch2_p2(route2)

        if self.stride > 1:
            output = torch.cat([route1, route2], 1)
        else:
            output = route1 + route2
        
        return output

    def channel_shuffle(self, x, groups):
        # Type: Tensor(N x Channel x Height, Width)
        _, channel, height, width = x.shape
        channel_per_groups = channel // groups
        x = x.view(_, groups, channel_per_groups, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(_, -1, height, width)

        return x

class ShuffleNet(nn.Module):

    groups_mapper = {1 : [144, 288, 576],
                     2 : [200, 400, 800],
                     3 : [240, 480, 960],
                     4 : [272, 544, 1088],
                     8 : [384, 768, 1536]}

    def __init__(self, groups):
        super(ShuffleNet, self).__init__()
        self.conv = nn.Conv2d(3, 24, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        out_channels = self.groups_mapper[groups]

        self.stage2 = self._make_stage(24, out_channels[0], 3, groups)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7, groups)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3, groups)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(out_channels[2], 1000)

    def forward(self, x):
        #Type: Tensor(N x channel x Height, Width)
        # N x 3 x 224 x 224
        x = self.conv(x)
        # N x 24 x 112 x 112
        x = self.maxpool(x)
        # N x 24 x 56 x 56
        x = self.stage2(x)
        # N x assgined_channel x 28 x 28
        x = self.stage3(x)
        # N x assigned_channel x 14 x 14
        x = self.stage4(x)
        # N x assigned_channel x 7 x 7
        x = self.pool(x)
        # N x assigned_channel x 1 x 1
        x = torch.flatten(x, 1)
        # N x assigned_channel
        x = self.fc(x)
        # N x 1000

        return x

    def _make_stage(self, inc, outc, times, groups):
        block = InvertedResidual
        holder = [block(inc, outc, 2, groups)]
        for _ in range(times):
            holder.append(block(outc, outc, 1, groups))
        
        return nn.Sequential(*holder)