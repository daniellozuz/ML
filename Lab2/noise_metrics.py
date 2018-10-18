import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


head_perfect = rgb2gray(plt.imread('1.jpg'))
head_disrupted = rgb2gray(plt.imread('3.jpg'))
print(head_perfect)
print(head_disrupted)
head_difference = abs(head_disrupted - head_perfect)


plt.subplot(311)
plt.imshow(head_perfect, cmap='gray')
plt.subplot(312)
plt.imshow(head_disrupted, cmap='gray')
plt.subplot(313)
plt.imshow(head_difference, cmap='gray')

print(head_difference)
PAE = np.amax(head_difference)
print('PAE:', PAE)
plt.show()
