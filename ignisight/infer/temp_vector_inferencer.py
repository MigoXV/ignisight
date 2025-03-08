import torch
from pathlib import Path
from ignisight.models import BertModel
from matplotlib import pyplot as plt
import numpy as np
import imageio
import torchvision

def ir_infer(model, x: torch.Tensor):
    x = x.mean(dim=1)
    seq_len = x.size(1)
    positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
    # token_emb = self.embedding(x)
    pos_emb = model.pos_embedding(positions)
    # Transformer Encoder
    x = model.encoder(x + pos_emb)  # [batch_size, seq_len, d_model]

    return x

class Inferencer:

    def __init__(self, ckpt_path):
        self.model = BertModel()
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt["model"])
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

    def infer(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
        return output.detach().cpu().numpy()



    def infer_from_path(self, image_path: Path) -> torch.Tensor:
        image = imageio.imread(image_path)
        # image = torch.from_numpy(image)
        image = self.transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            return ir_infer(self.model, image)


if __name__ == "__main__":
    inferencer = Inferencer(Path("outputs/test03/checkpoint_best.pt"))
    a = inferencer.infer_from_path(Path("data-bin/train01/images/202409011713.bmp"))
    a = a.squeeze(0).numpy()
    plt.imshow(a, cmap="gray")
    plt.show()
    plt.savefig("data-bin/cnn.png")
