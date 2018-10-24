import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.color import rgb2gray
from skimage.measure import compare_ssim


def mean_square_error(image1, image2):
    return np.sum(absdiff(image1, image2) ** 2) / image1.size

def peak_absolute_error(image1, image2):
    return np.amax(absdiff(image1, image2))

def absdiff(image1, image2):
    # Problem z underflow?
    print(image1[0][0])
    print(image2[0][0])
    print(image1[0][0] - image2[0][0])
    return np.fabs(image1 - image2).astype(np.uint8)

def structural_similarity_index(image1, image2):
    return compare_ssim(image1, image2, multichannel=True)

def peak_signal_to_noise_ratio(image1, image2):
    print()
    return 10 * math.log10(255 ** 2 / mean_square_error(image1, image2))


head_perfect = plt.imread('1.jpg').astype(np.uint8)
head_disrupted = plt.imread('3.jpg').astype(np.uint8)
heads = head_perfect, head_disrupted

plt.subplot(311)
plt.imshow(head_perfect, cmap='gray')
plt.subplot(312)
plt.imshow(head_disrupted, cmap='gray')
plt.subplot(313)
plt.imshow(absdiff(*heads), cmap='gray')

print('Peak Absolute Error:', peak_absolute_error(*heads))
print('Mean Square Error:', mean_square_error(*heads))
print('Peak Signal-to-Noise Ratio', peak_signal_to_noise_ratio(*heads))
print('Structural Similarity Index:', structural_similarity_index(*heads))

plt.show()
