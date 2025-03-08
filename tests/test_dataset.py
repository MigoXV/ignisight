import imageio
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

image_path = 'data-bin/temp_fix/images/202409011713.bmp'
image_path = Path(image_path)
image = imageio.imread(image_path)

temp = loadmat('data-bin/temp_fix/temp_mats/202409011713.mat')['thermalImage']
# Convert temperature data to Celsius if needed
# temp data is in Kelvin by default in many thermal cameras
# temp_celsius = temp - 273.15  # Uncomment if conversion is needed

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
ax1.imshow(image)
ax1.set_title('Original Image')
ax1.axis('off')

# Display the thermal image with plasma colormap
thermal_plot = ax2.imshow(temp, cmap='plasma')
ax2.set_title('Thermal Image')
ax2.axis('off')

# Add a color bar
cbar = fig.colorbar(thermal_plot, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Temperature (K)')

plt.tight_layout()
plt.show()