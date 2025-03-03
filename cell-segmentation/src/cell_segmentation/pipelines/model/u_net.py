import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=9, use_groupnorm=True, num_groups=8, upsample_mode="transposed"):

        super(UNet, self).__init__()

        self.use_groupnorm = use_groupnorm
        self.upsample_mode = upsample_mode
        self.num_groups = num_groups

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = self._conv_block(input_channels, 64, dropout=0.0, num_groups=num_groups)
        self.enc2 = self._conv_block(64, 128, dropout=0.1, num_groups=num_groups)
        self.enc3 = self._conv_block(128, 256, dropout=0.2, num_groups=num_groups)
        self.enc4 = self._conv_block(256, 512, dropout=0.3, num_groups=num_groups)

        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024, dropout=0.4, num_groups=num_groups)

        # Decoder
        self.up4 = self._upsample(1024, 512)
        self.dec4 = self._conv_block(1024, 512, dropout=0.3, num_groups=num_groups)

        self.up3 = self._upsample(512, 256)
        self.dec3 = self._conv_block(512, 256, dropout=0.2, num_groups=num_groups)

        self.up2 = self._upsample(256, 128)
        self.dec2 = self._conv_block(256, 128, dropout=0.1, num_groups=num_groups)

        self.up1 = self._upsample(128, 64)
        self.dec1 = self._conv_block(128, 64, dropout=0.0, num_groups=num_groups)

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x, apply_activation=False):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))
    
        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        outputs = self.final_conv(dec1)

        if apply_activation:
            if self.final_conv.out_channels == 1:
                outputs = torch.sigmoid(outputs)  # Binary segmentation
            else:
                outputs = torch.softmax(outputs, dim=1)  # Multi-class segmentation

        return outputs

    def _conv_block(self, in_channels, out_channels, dropout=0.0, num_groups=8):

        norm_layer = (
            nn.GroupNorm(num_groups, out_channels)
            if self.use_groupnorm
            else nn.BatchNorm2d(out_channels)
        )
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer,
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm_layer,
            nn.ReLU(inplace=True),
        )

    def _upsample(self, in_channels, out_channels):

        if self.upsample_mode == "transposed":
            # Use ConvTranspose2D for learnable upsampling
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            # Use bilinear interpolation
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        
    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
