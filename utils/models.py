import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class SeamRegressor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.backbone = models.shufflenet_v2_x1_0(pretrained=True)
        original_stem = self.backbone.conv1[0]
        self.backbone.conv1[0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=original_stem.bias
        )
        # with torch.no_grad():
        #     self.backbone.conv1[0].weight = nn.Parameter(
        #         original_stem.weight.mean(dim=1, keepdim=True)
        #     )
        num_features = self.backbone.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels)  # Выход: x и y координаты
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)
        x = self.backbone.conv5(x)
        x = torch.mean(x, dim=[-1, -2])  # globalpool
        x = self.fc(x)
        return x
        # return F.sigmoid(self.backbone(x))
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = F.relu(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, length = x.size()
        
        # Calculate queries, keys, values
        q = self.query(x).view(batch_size, -1, length)
        k = self.key(x).view(batch_size, -1, length)
        v = self.value(x).view(batch_size, -1, length)
        
        # Attention scores
        attention = torch.bmm(q.permute(0, 2, 1), k)
        attention = F.sigmoid(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, length)
        
        return self.gamma * out + x


class SeamRegressor1D(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Residual blocks with attention
        self.res_block1 = self._make_residual_block(64, 128, stride=2)
        self.attention1 = AttentionBlock(128)
        
        self.res_block2 = self._make_residual_block(128, 256, stride=2)
        self.attention2 = AttentionBlock(256)
        
        self.res_block3 = self._make_residual_block(256, 512, stride=2)
        
        # Additional conv layers
        self.conv_extra = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # Adaptive pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride, downsample),
            ResidualBlock(out_channels, out_channels)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.attention1(x)
        x = self.res_block2(x)
        x = self.attention2(x)
        x = self.res_block3(x)
        x = self.conv_extra(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SeamRegressorTiny(nn.Module):
    def __init__(self, input_height=64, input_width=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)  # 8x32x32
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1) # 16x16x16
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 32x8x8
        
        # Вычисляем размер фичей после сверток
        self.feature_size = self._get_conv_output(input_height, input_width)
        
        # Полносвязные слои для регрессии координат (x, y)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3))  # 2 выхода - x и y координаты
    
    def _get_conv_output(self, h, w):
        # Вспомогательная функция для вычисления размера фичей после сверток
        with torch.no_grad():
            x = torch.zeros(1, 1, h, w)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.view(1, -1).shape[1]
    
    def forward(self, x):
        # Применяем свертки с активациями
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Вытягиваем в вектор и применяем полносвязные слои
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(TinyUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)
        self.outc = OutConv(16, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
