from pathlib import Path
from typing import List

import imageio
import numpy as np
import torch
import torchvision
from fairseq.data import FairseqDataset
from scipy.io import loadmat


class TempFixDataset(FairseqDataset):

    def __init__(self, data_dir: Path, filenames: List[str]):
        super().__init__()
        self.data_dir = data_dir
        self.image_dir = data_dir / "images"
        self.temp_dir = data_dir / "temp_mats"
        self.filenames = filenames
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
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index] + ".bmp"
        image_path = self.image_dir / filename
        image = imageio.imread(image_path)
        image = self.transform(image).transpose(-1, -2)
        temp_mat_path = self.temp_dir / (self.filenames[index] + ".mat")
        temp_mat = loadmat(temp_mat_path)["thermalImage"]
        temp_mat = np.pad(temp_mat, ((0, 0), (0, 2)), "edge").astype(np.float32)
        temp_mat = (temp_mat - 500) / 500
        temp_mat = torch.tensor(temp_mat).unsqueeze(0).transpose(-1, -2)
        return image, temp_mat

    def size(self, indice):
        return 1

    def num_tokens(self, indice):
        return 1

    def collater(self, samples):
        images, temps = zip(*samples)
        images = torch.stack(images)
        temps = torch.stack(temps)
        return images, temps
