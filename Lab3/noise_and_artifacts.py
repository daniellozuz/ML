import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


head_disrupted = plt.imread('head.png').astype(np.float64)
lungs_disrupted = plt.imread('lungs.jpg').astype(np.float64)

print(head_disrupted.shape)
print(head_disrupted.dtype)
print(lungs_disrupted.shape)
print(lungs_disrupted.dtype)


print(np.all(head_disrupted[:,:,0] == head_disrupted[:,:,1]))

print(head_disrupted[243,321,2])

plt.subplot(221)
plt.imshow(head_disrupted, cmap='gray')
plt.subplot(222)
plt.imshow(lungs_disrupted, cmap='gray')
# plt.subplot(421)
# plt.imshow(absdiff(*heads), cmap='gray')

plt.show()
