from pathlib import Path

import imageio
import torch
import torchvision
from matplotlib import pyplot as plt

from ignisight.models import DenoisingUNet


class Inferencer:

    def __init__(self, ckpt_path, device="cpu"):
        self.device = device
        self.model = DenoisingUNet()
        ckpt = torch.load(ckpt_path,weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.to(device)
        self.model.eval()
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
        image = self.transform(image).to(self.device)
        image = image.transpose(-1, -2).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image) * 500 + 500
        return output.cpu().numpy()

    def infer_from_path(self, image_path: Path) -> torch.Tensor:
        image = imageio.imread(image_path)
        return self.infer(image)


if __name__ == "__main__":
    inferencer = Inferencer(Path("outputs/temp_fix/test01/checkpoint_best.pt"))
    image = imageio.imread("data-bin/train01/images/202409011713.bmp")
    # a = inferencer.infer_from_path(Path("data-bin/train01/images/202409011713.bmp"))
    temp = inferencer.infer(image)
    temp = temp.squeeze(0).swapaxes(0, 2)
    # plt.imshow(a, cmap="gray")
    # plt.show()
    # plt.savefig("data-bin/unet.png")

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Display the thermal image with plasma colormap
    thermal_plot = ax2.imshow(temp, cmap="plasma")
    ax2.set_title("Thermal Image")
    ax2.axis("off")

    # Add a color bar
    cbar = fig.colorbar(thermal_plot, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (K)")

    plt.tight_layout()
    plt.show()
