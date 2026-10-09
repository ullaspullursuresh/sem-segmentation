import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionDown, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return self.pool(out), out

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.upConv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip):
        out = self.upConv(x)
        if out.size() != skip.size():
            out = F.interpolate(out, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=True)
        return torch.cat([out, skip], 1)

class DenseUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=48, growth_rate=16, block_config=(4, 4, 4, 4)):
        super(DenseUNet, self).__init__()
        
        self.conv_init = nn.Conv2d(in_channels, init_features, kernel_size=3, padding=1, bias=False)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.trans_down = nn.ModuleList()
        num_features = init_features
        self.skip_channels = [] # Explicitly track skip connection sizes
        
        for num_layers in block_config:
            self.encoder_blocks.append(DenseBlock(num_layers, num_features, growth_rate))
            num_features += num_layers * growth_rate
            self.trans_down.append(TransitionDown(num_features, num_features))
            self.skip_channels.append(num_features) # Save exact channel count for the decoder
            
        # Bottleneck
        self.bottleneck = DenseBlock(4, num_features, growth_rate)
        num_features += 4 * growth_rate
        
        # Decoder
        self.trans_up = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i, num_layers in enumerate(reversed(block_config)):
            skip_features = self.skip_channels[-(i+1)] # Retrieve the exact matching skip size
            self.trans_up.append(TransitionUp(num_features, skip_features))
            num_features = skip_features * 2 # TransitionUp output concatenated with skip connection
            self.decoder_blocks.append(DenseBlock(num_layers, num_features, growth_rate))
            num_features += num_layers * growth_rate
            
        self.final_bn = nn.BatchNorm2d(num_features)
        self.final_conv = nn.Conv2d(num_features, out_channels, kernel_size=1)

    def forward(self, x):
        features = self.conv_init(x)
        skip_connections = []
        
        for enc_block, trans in zip(self.encoder_blocks, self.trans_down):
            features = enc_block(features)
            features, skip = trans(features)
            skip_connections.append(skip)
            
        features = self.bottleneck(features)
        
        for dec_block, trans_up, skip in zip(self.decoder_blocks, self.trans_up, reversed(skip_connections)):
            features = trans_up(features, skip)
            features = dec_block(features)
            
        features = self.final_conv(F.relu(self.final_bn(features)))
        return torch.sigmoid(features)