from pathlib import Path
from typing import List

import imageio
import pandas as pd
import torch
import torchvision
from fairseq.data import FairseqDataset
from PIL import Image


class IgnisightDataset(FairseqDataset):

    def __init__(self, image_dir: Path, xls_path: Path):
        self.image_dir = image_dir
        self.xls_path = xls_path
        self.df = pd.read_excel(xls_path)
        self.df.columns = [
            "Time",
            "IR_Upper_TC",
            "IR_Left2_TC",
            "IR_SiC_Upper",
            "Temp_Set",
            "Left2_TC",
            "Upper_TC",
            "SiC_Upper_Actual",
        ]
        # 形状固定为384x288
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((288, 384)),
                torchvision.transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def size(self, indice):
        return 1

    def num_tokens(self, indice):
        return 1

    def __getitem__(self, indice):
        time_stamp = self.df.iloc[indice]["Time"]
        image_path = time_stamp.strftime(r"%Y%m%d%H%M")
        image_path = self.image_dir / (image_path + ".bmp")
        image = imageio.imread(image_path)
        # image = torch.from_numpy(image).permute(2,1,0)
        image = self.transform(image).transpose(-1, -2)
        tgt_vector = torch.Tensor(
            self.df.drop(columns=["Time", "Temp_Set"]).iloc[indice].values
        )/1000
        return image, tgt_vector

    def collater(self, samples: List[torch.Tensor]):
        images, tgt_vectors = list(zip(*samples))
        images = torch.stack(images)
        tgt_vectors = torch.stack(tgt_vectors)
        return images, tgt_vectors


if __name__ == "__main__":
    image_dir = Path("data-bin/train01/images")
    xls_path = Path("data-bin/train01/红外数据整理.xls")
    ignisight_dataset = IgnisightDataset(image_dir, xls_path)
    # print(ignisight_dataset[0])
    # print(ignisight_dataset[0][0].shape)
    # print(ignisight_dataset[0][1].shape)
    # print(ignisight_dataset.collater([ignisight_dataset[0], ignisight_dataset[1]]))

    # 随机抽取一个样本
    import random

    import matplotlib.pyplot as plt

    random_index = random.randint(0, len(ignisight_dataset))
    image, tgt_vector = ignisight_dataset[random_index]
    plt.imshow(image.squeeze(0).permute(2, 1, 0).numpy())
    plt.show()
    print(tgt_vector)
    print(tgt_vector.shape)
