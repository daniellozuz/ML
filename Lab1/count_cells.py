import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage.measurements import label


TRESHOLD = 190
ROI = np.s_[600:700, 600:700]


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]).astype(int)


blood_cells = plt.imread('blood_cells.jpg')
blood_cells_gray = rgb2gray(blood_cells)
plt.subplot(221)
plt.title('Grayscale')
plt.imshow(blood_cells_gray[ROI], cmap='gray')

blood_cells_binary = blood_cells_gray < TRESHOLD
_, num_features = label(blood_cells_binary)
plt.subplot(222)
plt.title(f'Binarized. Cells: {num_features}')
plt.imshow(blood_cells_binary[ROI], cmap='gray')

blood_cells_binary = binary_erosion(blood_cells_binary)
_, num_features = label(blood_cells_binary)
plt.subplot(223)
plt.title(f'Eroded. Cells: {num_features}')
plt.imshow(blood_cells_binary[ROI], cmap='gray')

blood_cells_binary = binary_dilation(blood_cells_binary)
_, num_features = label(blood_cells_binary)
plt.subplot(224)
plt.title(f'Dilated. Cells: {num_features}')
plt.imshow(blood_cells_binary[ROI], cmap='gray')

plt.show()
