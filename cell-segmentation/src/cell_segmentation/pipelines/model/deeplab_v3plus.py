import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPP(nn.Module):
    """ Atrous Spatial Pyramid Pooling (ASPP) module for multi-scale feature extraction """

    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0], bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1], bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2], bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.conv_final = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))
        x4 = F.relu(self.bn4(self.conv4(x)))

        x5 = self.global_avg_pool(x)
        x5 = F.relu(self.bn5(self.conv5(x5)))
        x5 = F.interpolate(x5, size=x.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = F.relu(self.bn_final(self.conv_final(x)))
        return x


class DeepLabV3Plus(nn.Module):
    """ DeepLabV3+ model with ResNet50 as a backbone """

    def __init__(self, num_classes=9, backbone="resnet50", pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            low_level_channels = 256
            aspp_channels = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            low_level_channels = 256
            aspp_channels = 2048
        else:
            raise ValueError("Backbone must be 'resnet50' or 'resnet101'.")

        # Remove fully connected layers
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # ASPP Module
        self.aspp = ASPP(aspp_channels, out_channels=256)

        # Decoder
        self.decoder_conv1 = nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False)
        self.decoder_bn1 = nn.BatchNorm2d(48)

        self.decoder_conv2 = nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False)
        self.decoder_bn2 = nn.BatchNorm2d(256)

        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        # Extract features
        low_level_features = self.backbone[0:5](x)  # Shape: (B, 256, H/4, W/4)
        high_level_features = self.backbone(x)  # Shape: (B, 2048, H/16, W/16)

        # ASPP
        high_level_features = self.aspp(high_level_features)  # Shape: (B, 256, H/16, W/16)

        # Upsample high-level features
        high_level_features = F.interpolate(high_level_features, size=low_level_features.shape[2:], mode="bilinear", align_corners=False)

        # Process low-level features
        low_level_features = F.relu(self.decoder_bn1(self.decoder_conv1(low_level_features)))

        # Concatenate features
        x = torch.cat([high_level_features, low_level_features], dim=1)

        # Decoder
        x = F.relu(self.decoder_bn2(self.decoder_conv2(x)))
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        # Final classification layer
        x = self.final_conv(x)
        return x
