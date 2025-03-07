import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def temp_correction(tempData):
    tempData = (
        -7.8836 * 10**-6 * tempData**3
        + 0.0176 * tempData**2
        - 10.815 * tempData
        + 2725.9628
    )
    return tempData


HEIGHT = 288
WIDTH = 384


def draw_depth():
    data = np.genfromtxt("data-bin/img_dep_384x288.csv", delimiter=",").astype(
        np.float32
    )
    x1 = np.repeat(np.arange(1, HEIGHT + 1), WIDTH)
    y1 = np.tile(np.arange(1, WIDTH + 1), HEIGHT)
    z1 = data[1:, 1:].flatten()

    xi = np.linspace(1, HEIGHT)
    yi = np.linspace(1, WIDTH)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata(np.vstack((x1, y1)).T, z1, (xi, yi), method="cubic")

    fig1 = plt.figure(1)
    ax = plt.axes(projection="3d")
    surf = ax.plot_surface(xi, yi, zi, cmap="BuPu", linewidth=0, antialiased=False)
    fig1.colorbar(surf)
    ax.set_title("深度图")
    plt.xlabel("x")
    plt.show()


def draw_temperature():
    new_temp_correction = (
        lambda x: -7.8836 * 10**-6 * x**3 + 0.0176 * x**2 - 10.815 * x + 2725.9628
    )
    # data = scio.loadmat("data-bin/4时58分.mat")
    data = scio.loadmat("tmp-workspace/匣钵区域温度校正/第一组(240901-240902)/温度矩阵/202409011713.mat")
    temp_data = np.array(data["thermalImage"])  # 将matlab数据赋值给python变量
    # old_temp_data=temp_correction(temp_data)
    new_temp_data = new_temp_correction(temp_data)
    data_point = np.ndarray(shape=(288 * 382, 3), dtype=float, order="F")
    # print(temp_data)
    temp_data2 = np.zeros_like(temp_data,dtype=float)
    for i in range(288):
        for j in range(382):
            temp_data2[i, j] = temp_correction(temp_data[i, j])
            data_point[i * 382 + j, 0] = i
            data_point[i * 382 + j, 1] = j
            data_point[i * 382 + j, 2] = temp_data2[i, j]
    x1 = np.repeat(np.arange(1, 288 + 1), 382)
    y1 = np.tile(np.arange(1, 382 + 1), 288)
    x = data_point[:, 0]
    y = data_point[:, 1]
    z = data_point[:, 2]
    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata(data_point[:, 0:2], z, (xi, yi), method="cubic")
    # Create a heatmap of the temperature data
    fig1=plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(temp_data, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap, label='Temperature')
    plt.title('Temperature Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('temp_heatmap.png')
    # plt.show()

    # Also create a heatmap for the corrected temperature data
    fig2=plt.figure(figsize=(10, 8))
    heatmap_corrected = plt.imshow(temp_data2, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap_corrected, label='Corrected Temperature')
    plt.title('Corrected Temperature Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('corrected_temp_heatmap.png')
    # plt.show()
    fig3 = plt.figure(3)
    ax = plt.axes(projection="3d")
    surf = ax.plot_surface(xi, yi, zi, cmap="BuPu", linewidth=0, antialiased=False)
    pickle.dump((xi, yi, zi), open("data-bin/raw_temp.pkl", "wb"))
    fig3.colorbar(surf)
    ax.set_title("温度图")
    plt.show()


if __name__ == "__main__":
    # draw_depth()
    draw_temperature()
