import pickle
import numpy as np

raw_temp = pickle.load(open("data-bin/raw_temp.pkl", "rb"))
raw_xi, raw_yi, raw_zi = raw_temp
temp = pickle.load(open("data-bin/temp.pkl", "rb"))
xi, yi, zi = temp


assert raw_xi.shape == xi.shape and raw_yi.shape == yi.shape and raw_zi.shape == zi.shape
mse_x = np.mean(np.abs(raw_xi - xi))
mse_y = np.mean(np.abs(raw_yi - yi))
mse_z = np.mean(np.abs(raw_zi - zi))
print(f"xi: {mse_x}, yi: {mse_y}, zi: {mse_z}")
assert mse_x < 1e-5 and mse_y < 1e-5 and mse_z < 1e-5
