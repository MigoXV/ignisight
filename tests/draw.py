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




x1 = []
y1 = []
z1 = []
data = np.genfromtxt("data-bin/img_dep_384x288.csv", delimiter=",").astype(np.float32)
dataPoint = np.ndarray(shape=(288 * 384, 3), dtype=float, order="F")
for i in range(288):
    for j in range(384):
        x1.append(i + 1)
        y1.append(j + 1)
        z1.append(data[i + 1, j + 1])
for i in range(288 * 384):
    dataPoint[i, 0] = x1[i]
    dataPoint[i, 1] = y1[i]
    dataPoint[i, 2] = z1[i]

x = dataPoint[:, 0]
y = dataPoint[:, 1]
z = dataPoint[:, 2]
xi = np.linspace(min(x), max(x))
yi = np.linspace(min(y), max(y))
xi, yi = np.meshgrid(xi, yi)
zi = griddata(dataPoint[:, 0:2], z, (xi, yi), method="cubic")

fig1 = plt.figure(1)
ax = plt.axes(projection="3d")
pickle.dump((xi, yi, zi), open("data-bin/raw_dep.pkl", "wb"))
surf = ax.plot_surface(xi, yi, zi, cmap="BuPu", linewidth=0, antialiased=False)
fig1.colorbar(surf)
ax.set_title("深度图")
plt.xlabel("x")


data = scio.loadmat("data-bin/tempData.mat")
tempData = np.array(data["thermalImage"])  # 将matlab数据赋值给python变量
dataPoint = np.ndarray(shape=(288 * 382, 3), dtype=float, order="F")
print(tempData)
for i in range(288):
    for j in range(382):
        tempData[i, j] = temp_correction(tempData[i, j])
        dataPoint[i * 382 + j, 0] = i
        dataPoint[i * 382 + j, 1] = j
        dataPoint[i * 382 + j, 2] = tempData[i, j]
x = dataPoint[:, 0]
y = dataPoint[:, 1]
z = dataPoint[:, 2]
xi = np.linspace(min(x), max(x))
yi = np.linspace(min(y), max(y))
xi, yi = np.meshgrid(xi, yi)
zi = griddata(dataPoint[:, 0:2], z, (xi, yi), method="cubic")


fig2 = plt.figure(2)
ax = plt.axes(projection="3d")
surf = ax.plot_surface(xi, yi, zi, cmap="BuPu", linewidth=0, antialiased=False)
pickle.dump((xi, yi, zi), open("data-bin/raw_temp.pkl", "wb"))
fig2.colorbar(surf)
ax.set_title("温度图")
plt.show()
