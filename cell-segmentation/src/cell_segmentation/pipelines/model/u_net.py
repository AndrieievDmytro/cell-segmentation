import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=8):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self._conv_block(input_channels, 64, dropout=0.1)
        self.enc2 = self._conv_block(64, 128, dropout=0.1)
        self.enc3 = self._conv_block(128, 256, dropout=0.2)
        self.enc4 = self._conv_block(256, 512, dropout=0.2)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024, dropout=0.3)

        # Decoder
        self.dec4 = self._conv_block(1024 + 512, 512, dropout=0.2)
        self.dec3 = self._conv_block(512 + 256, 256, dropout=0.2)
        self.dec2 = self._conv_block(256 + 128, 128, dropout=0.1)
        self.dec1 = self._conv_block(128 + 64, 64, dropout=0.1)

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _conv_block(self, in_channels, out_channels, dropout):
        """Helper function to create a convolutional block with Conv2D, Dropout, and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        # Decoder
        dec4 = self.dec4(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=True), enc4], dim=1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, scale_factor=2, mode="bilinear", align_corners=True), enc3], dim=1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, scale_factor=2, mode="bilinear", align_corners=True), enc2], dim=1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, scale_factor=2, mode="bilinear", align_corners=True), enc1], dim=1))

        # Final layer
        outputs = self.final_conv(dec1)
        return outputs
