import numpy as np
import scipy.io as scio

from ignisight.draw.cloud_map import TemperatureCloudVisualizer


def get_temp():

    base_data = scio.loadmat(
        "tmp-workspace/匣钵区域温度校正/第一组(240901-240902)/温度矩阵/202409011713.mat"
    )
    base_data = np.array(base_data["thermalImage"])
    # 对温度数据进行边缘填充，保持数据尺寸一致
    base_data = np.pad(base_data, ((0, 0), (0, 2)), "edge")
    counter = 0

    def get_temperature_data() -> np.ndarray:
        """
        模拟接收新的温度数据。
        实际使用时，可在此方法中添加数据采集逻辑（例如从传感器或网络接收数据）。
        此处模拟一个 (250, 224) 的温度矩阵，数据会随时间变化。
        """
        nonlocal counter, base_data
        # 利用正弦波制造周期性温度波动
        variation = 100 * np.sin(counter / 100.0)
        noise = np.random.randn(288, 384) * 100
        tempData = variation + noise + base_data
        counter += 1
        return tempData

    return get_temperature_data


# 示例用法：启动实时温度云图（这里不再从文件加载 mat 数据，而是使用定时器模拟实时数据更新）
if __name__ == "__main__":
    visualizer = TemperatureCloudVisualizer()
    visualizer.run(get_temp())
