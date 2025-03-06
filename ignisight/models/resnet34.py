import torch
import torch.nn as nn
import torchvision.models as models
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture


@register_model("dilated_resnet")
class DilatedResNet34(BaseFairseqModel):

    @classmethod
    def build_model(cls, args, task):
        return cls()

    def __init__(self, num_outputs=6):
        super(DilatedResNet34, self).__init__()
        # 使用 replace_stride_with_dilation 参数，对 layer3 和 layer4 使用膨胀卷积
        self.base = models.resnet34(
            pretrained=False
        )
        # 修改最后的全连接层，使输出为6维向量
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.base(x)


@register_model_architecture("dilated_resnet", "dilated_resnet34")
def register_dilated_resnet34_model(args):
    pass


# 测试模型
if __name__ == "__main__":
    model = DilatedResNet34(num_outputs=6)
    # 构造一个 batch 大小为 8，输入形状为 (8, 3, 384, 288) 的随机样本
    x = torch.randn(8, 3, 384, 288)
    output = model(x)
    print("输出形状:", output.shape)  # 应输出 torch.Size([8, 6])
