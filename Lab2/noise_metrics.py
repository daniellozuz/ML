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
    return np.fabs(image1.astype(np.int16) - image2.astype(np.int16)).astype(np.uint8)


def structural_similarity_index(image1, image2):
    return compare_ssim(image1, image2, multichannel=True)


def peak_signal_to_noise_ratio(image1, image2):
    return 10 * math.log10(255 ** 2 / mean_square_error(image1, image2))


def gaussian_noise(image, mean, std):
    image_noisy = image.astype(np.int16) + np.random.normal(mean, std, image.shape).astype(np.int16)
    return np.clip(image_noisy, 0, 255).astype(np.uint8)


def quality_metrics(image1, image2):
    images = image1, image2
    return {
        'Peak Absolute Error': peak_absolute_error(*images),
        'Mean Square Error': mean_square_error(*images),
        'Peak Signal-to-Noise Ratio': peak_signal_to_noise_ratio(*images),
        'Structural Similarity Index': structural_similarity_index(*images),
    }


def ex1_1():
    head_perfect = plt.imread('1.jpg').astype(np.uint8)
    head_disrupted = plt.imread('3.jpg').astype(np.uint8)
    heads = head_perfect, head_disrupted
    plt.subplot(311)
    plt.imshow(head_perfect, cmap='gray')
    plt.subplot(312)
    plt.imshow(head_disrupted, cmap='gray')
    plt.subplot(313)
    plt.imshow(absdiff(*heads), cmap='gray')
    print(quality_metrics(*heads))
    plt.show()


def ex1_2():
    mean = 0.0
    stds = [5.0, 30.0, 100.0, 500.0]
    head_perfect = plt.imread('1.jpg').astype(np.uint8)
    for n, std in enumerate(stds):
        head_noisy = gaussian_noise(head_perfect, mean, std)
        plt.subplot(220 + n + 1)
        plt.imshow(head_noisy)
        print(quality_metrics(head_perfect, head_noisy))
    plt.show()


def ex2():
    raise 'Hard to do in Python, consider using matlab.engine'


if __name__ == '__main__':
    ex1_2()
