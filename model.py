"""CIFAR-adapted 18-layer residual network; state keys match the executed notebook."""
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout))
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.branch(x) + self.shortcut(x))


class SmallResNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        blocks, cin = [], width
        for i, cout in enumerate([width, width*2, width*4, width*8]):
            blocks.extend([BasicBlock(cin, cout, 1 if i == 0 else 2), BasicBlock(cout, cout)])
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.fc(self.pool(self.blocks(self.stem(x))).flatten(1))
