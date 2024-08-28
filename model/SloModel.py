from torch import nn
import torch
import torch.nn.functional as F
class SloNet(nn.Module):
    def __init__(self, cfgs):
        super(SloNet, self).__init__()
        dim = cfgs['model_cfg']['incomplete_fusion']['dim']
        classes = cfgs['model_cfg']['BCVA_Num_Classes']
        self.isMSA = cfgs['base_cfg']['isMSA']
        self.backbone = nn.Sequential(nn.Conv2d(3, 64, kernel_size = (7,7), stride = (2,2), padding = 3),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),

                                      )
        if self.isMSA:
            self.MSA1 = MultiScaleAttention(64)
            self.block1 = Block(64, 128)
            self.conv1 = nn.Conv2d(64, 128, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1))

            self.MSA2 = MultiScaleAttention(128)
            self.block2 = Block(128, 256)
            self.conv2 = nn.Conv2d(128, 256, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1))

            self.MSA3 = MultiScaleAttention(256)
            self.block3 = Block(256, dim, isBottleneck = False)
            self.conv3 = nn.Conv2d(256, dim, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1))
        else:
            self.block1 = Block(64, 128)
            self.block2 = Block(128, 256)
            self.block3 = Block(256, dim, isBottleneck = False)

        self.pred = nn.Sequential(nn.LayerNorm(dim),
                                  nn.ReLU(),
                                  nn.Dropout(p = 0.5),
                                  nn.Linear(dim, classes),
                                  )
    def forward(self, x):
        if self.isMSA:
            x1 = self.backbone(x) # [b, 64, 56, 56]
            x2 = self.MSA1(x1)
            x3 = self.block1(x2) + self.conv1(x1) # [b, 128, 28, 28]

            x4 = self.MSA2(x3)
            x5 = self.block2(x4) + self.conv2(x3) # [b, 256, 14, 14]
            x6 = self.MSA3(x5)
            x7 = self.block3(x6) + self.conv3(x5) # [b, 384, 7, 7]
        else:
            x1 = self.backbone(x)  # [b, 64, 56, 56]
            x3 = self.block1(x1)
            x5 = self.block2(x3)
            x7 = self.block3(x5)

        embed = x7.flatten(start_dim = 2).permute(0, 2, 1) # [b, 49, 384]
        predSlo = self.pred(F.adaptive_avg_pool2d(x7,(1,1)).squeeze())
        return embed, predSlo.squeeze()

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, isBottleneck = True):
        super(Block, self).__init__()
        self.isBottleneck = isBottleneck
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1)),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(),
                                   )
        self.downSample = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size = (1, 1), stride = (2, 2)),
                                        nn.BatchNorm2d(out_channel),
                                        nn.ReLU(),
                                        )
        if isBottleneck:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel // 4, kernel_size = (1, 1), stride = (1, 1)),
                nn.BatchNorm2d(out_channel // 4),
                nn.ReLU(),
                nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.BatchNorm2d(out_channel // 4),
                nn.ReLU(),
                nn.Conv2d(out_channel // 4, out_channel, kernel_size = (1, 1), stride = (1, 1)),
                nn.BatchNorm2d(out_channel),
                )
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        res = self.downSample(identity)
        x = x + res
        if self.isBottleneck:
            x = self.bottleneck(x)
        return x

class MultiScaleAttention(nn.Module):
    def __init__(self, in_dim):
        super(MultiScaleAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size = (1,1))
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size = (1,1))
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = (1,1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, w * h)

        energy = torch.bmm(proj_query, proj_key)  # batch matmul
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(b, -1, w * h)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        out = self.gamma * out + x
        return out