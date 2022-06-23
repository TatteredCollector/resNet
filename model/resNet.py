import torch
import torch.nn as nn
from torchsummary import summary

'''
方差 标准差是方差的平方根
BN 批归一化: 网络比较深的时候 底部参数变化缓慢，底部变化快  
固定mini_batch每层输出的整体分布：x(1+i)=γ(xi-μ)/σ + β
（计算出小批量的均值和方差 然后学习出合适的便宜和缩放）  
线性变化  
全连接层：作用在特征维度（ dim=0 按行操作 求出每一列的 Nxfc ->1xN），
卷积层：作用在通道维(dim = 0,2,3 N*C*H*W ->1*N*1*1)
加速收敛速度 一般不改变模型精度
非线性激活 
二者之间的顺序：BN在激活之前 

卷积之后，如果要接BN操作，最好是不设置偏置，因为不起作用，而且占显卡内存。

'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel,
                 stride=1, downsample=None, **kwargs):
        # 这里的in——channel
        # 和out_channel是指残差块的整体输入输出通道数
        super(BasicBlock, self).__init__()
        # 第一个卷积 resnet一共四层layer 虚线残差块里（第一个残差块） 要进行下采样，所以步长为2
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    resnet 与resnext 十分相似，就是基本模块里面的前两层的输出通道数不同
    引入的基数和分组深度 例如resnet50_32*4d layer0中 resnet:64
    resnext:先将卷积通道缩放到128，然后进行分组卷积（例如第一层 128分为32组）
    分组卷积的可学习参数共享，来减少参数量：3*3*128*4
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1,
                 downsample=None, groups=1, width_per_group=64):
        # 这里的in——channel 和out_channel是指第一个主干卷积的参数
        super(Bottleneck, self).__init__()

        width = int(out_channel * width_per_group / 64.0) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------
        self.relu = nn.ReLU(inplace=True)
        # ------------------------------
        # 第二个卷积在（resnet一共四层 虚线残差块里）要进行下采样，
        # 所以步长可变 同时要进行分组卷积
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        # -------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=self.expansion * out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channel)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # 第三次卷积归一化之后 先残差链接再激活
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000,
                 include_top=True, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.groups = groups
        self.width_per_group = width_per_group
        self.in_channel = 64
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, blocks_num[0], 64)
        self.layer2 = self.make_layer(block, blocks_num[1], 128, stride=2)
        self.layer3 = self.make_layer(block, blocks_num[2], 256, stride=2)
        self.layer4 = self.make_layer(block, blocks_num[3], 512, stride=2)
        if self.include_top:
            self.averg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.flatten = nn.Flatten(start_dim=1)
            self.fc = nn.Linear(in_features=512 * block.expansion,
                                out_features=num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    def make_layer(self, block, block_num, channel, stride=1):
        downsample = None
        if stride != 1 or channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layer = []
        layer.append(block(
            in_channel=self.in_channel,
            out_channel=channel,
            stride=stride,
            downsample=downsample,
            groups=self.groups,
            width_per_group=self.width_per_group
        ))
        self.in_channel = channel * block.expansion
        # print(self.in_channel)
        for _ in range(1, block_num):
            layer.append(block(
                in_channel=self.in_channel,
                out_channel=channel,
                groups=self.groups,
                width_per_group=self.width_per_group
            ))
            # 实参——如果*号加在了是实参上，代表的是将输入迭代器拆成一个个元素。
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.averg_pool(x)
            x = self.flatten(x)
            x = self.fc(x)
        return x


def resnet34(num_classes=1000, include_top=True):
    blocks_num = [3, 4, 6, 3]
    return ResNet(BasicBlock, blocks_num, num_classes, include_top)


def resnet50(num_classes=1000, include_top=True):
    blocks_num = [3, 4, 6, 3]
    return ResNet(Bottleneck, blocks_num, num_classes, include_top)


def resnet101(num_classes=1000, include_top=True):
    blocks_num = [3, 4, 23, 3]
    return ResNet(Bottleneck, blocks_num, num_classes, include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    blocks_num = [3, 4, 6, 3]
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, blocks_num, num_classes, include_top, groups, width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    blocks_num = [3, 4, 23, 3]
    groups = 32
    width_per_group = 8
    return ResNet(block=Bottleneck, blocks_num=blocks_num,
                  num_classes=num_classes, include_top=include_top,
                  groups=groups, width_per_group=width_per_group)


if __name__ == "__main__":
    net = resnext101_32x8d()
    net.to(torch.device("cuda:0"))

    print(net)
    summary(net, input_size=(3, 224, 224))
