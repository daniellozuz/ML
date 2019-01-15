import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import watershed
from skimage.color import rgb2gray
from scipy.ndimage.morphology import binary_fill_holes


def jcet():
    folder = 'zajecia/zajecia'
    image_names = [
        'PFDS17_004W_1744M2B101P_HE_20X_3',
        'PFDS17_004W_1744M2B101P_HE_20X_4',
        'PFDS17_004W_1744M2B101P_HE_20X_5',
        'PFDS17_004W_1744M2B101P_HE_20X_6',
        'PFDS17_004W_1744M2B205P_HE_20X_1',
        'PFDS17_004W_1744M2B205P_HE_20X_2',
    ]
    images = [os.path.join(folder, im) for im in image_names]
    plt.figure()
    for n, image in enumerate(sorted(images)):
        lungs = plt.imread(image + ' ZAZN.png', 0)
        lungs.setflags(write=1)
        plt.subplot(3, 6, n + 1)
        plt.imshow(lungs)
        contour = get_contour(lungs)
        plt.subplot(3, 6, n + 7)
        plt.imshow(contour)
        flooded = flood(lungs, contour)
        plt.subplot(3, 6, n + 13)
        plt.imshow(flooded)
    plt.show()


def get_contour(image):
    return image[:, :, 0] == 255


def flood(image, contour):
    marker = np.zeros_like(contour)
    marker[1:10, 1:10] = 2
    marker[440, 440] = 1
    print(marker)
    print(contour)
    image[contour] = 255
    # return watershed(rgb2gray(image), marker)
    # return rgb2gray(image)
    return binary_fill_holes(contour)

if __name__ == '__main__':
    jcet()
