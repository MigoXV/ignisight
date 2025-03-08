from pathlib import Path

import imageio
import numpy as np
import scipy.io as scio

from ignisight.draw.cloud_map import TemperatureCloudVisualizer
from ignisight.infer.temp_fix import Inferencer


def get_temp():

    inferencer = Inferencer(
        Path("outputs/temp_fix/test01/checkpoint_best.pt"), device="cuda"
    )
    image_dir = Path("data-bin/temp_fix/images")
    image_list = list(image_dir.glob("*.bmp"))
    counter = 0

    def get_temperature_data() -> np.ndarray:
        """
        模拟接收新的温度数据。
        实际使用时，可在此方法中添加数据采集逻辑（例如从传感器或网络接收数据）。
        此处模拟一个 (250, 224) 的温度矩阵，数据会随时间变化。
        """
        nonlocal counter, inferencer, image_list
        image = imageio.v2.imread(image_list[counter % len(image_list)])
        temp = inferencer.infer(image)
        temp = temp[0, 0, :, :]
        counter += 1
        return temp

    return get_temperature_data


# 示例用法：启动实时温度云图（这里不再从文件加载 mat 数据，而是使用定时器模拟实时数据更新）
if __name__ == "__main__":
    visualizer = TemperatureCloudVisualizer()
    visualizer.run(get_temp())
