import torch
import torch.nn as nn
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture


@register_model("unet")
class DenoisingUNet(BaseFairseqModel):
    @classmethod
    def build_model(cls, args, task):
        return cls()

    def __init__(self):
        super(DenoisingUNet, self).__init__()

        # 编码器
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        # 中间层
        self.middle = self.conv_block(128, 256, pool=False)  # 最后一层不需要池化
        # 解码器
        self.dec3 = self.upconv_block(256 + 128, 128)
        self.dec2 = self.upconv_block(128 + 64, 64)
        # 修改 dec1：在最后上采样后增加一层卷积层强化输出
        self.dec1 = nn.Sequential(
            self.upconv_block(64 + 32, 32 + 16, final_activation=False),
            nn.Conv2d(32 + 16, 1, kernel_size=3, padding=1),
        )

    def conv_block(self, in_channels, out_channels, pool=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))  # 非中间层使用池化
        return nn.Sequential(*layers)

    def upconv_block(self, in_channels, out_channels, final_activation=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        ]
        if final_activation:
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 编码过程，同时保存各层的特征用于跳跃连接
        enc1_out = self.enc1(x)  # (B, 64, 384, 288)
        enc2_out = self.enc2(enc1_out)  # (B, 128, 192, 144)
        enc3_out = self.enc3(enc2_out)  # (B, 256, 96, 72)
        # 中间层
        middle_out = self.middle(enc3_out)  # (B, 1024, 48, 36)
        # 解码过程，利用跳跃连接
        dec3_out = self.dec3(
            torch.cat([middle_out, enc3_out], dim=1)
        )  # (B, 256, 192, 144)
        dec2_out = self.dec2(
            torch.cat([dec3_out, enc2_out], dim=1)
        )  # (B, 128, 384, 288)
        dec1_out = self.dec1(torch.cat([dec2_out, enc1_out], dim=1))  # (B, 1, 384, 288)

        return dec1_out


@register_model_architecture("unet", "base_unet")
def base_unet(args):
    pass
