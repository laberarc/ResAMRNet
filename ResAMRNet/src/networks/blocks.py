import torch.nn as nn


def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channel, out_channel, stride=1,padding=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=False)


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x) #bs,c,1,1
        avg_out = avg_out.squeeze(-1).permute(0,2,1) #bs,1,c
        avg_out = self.conv(avg_out) #bs,1,c
        max_out = self.max_pool(x)  # bs,c,1,1
        max_out = max_out.squeeze(-1).permute(0, 2, 1)  # bs,1,c
        max_out = self.conv(max_out)  # bs,1,c
        y = avg_out + max_out
        y = self.sigmoid(y) #bs,1,c
        y = y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, padding=1,downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride,padding)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.eca = ECAAttention(kernel_size=5)
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        eca_out = self.eca(out)
        out = self.relu(self.downsample(x) + out+ eca_out)

        return out

class ResBlockSp(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, padding=1,downsample=None):
        super(ResBlockSp, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride,padding)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.eca = ECAAttention(kernel_size=5)
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        eca_out = self.eca(out)
        # out = self.relu(self.downsample(x) + out+ eca_out)
        return out,self.downsample(x),eca_out

class ResBlock1x1(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock1x1, self).__init__()
        self.conv1 = conv1x1(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.eca = ECAAttention(kernel_size=5)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        eca_out = self.eca(out)
        out = self.relu(x + out + eca_out)

        return out
