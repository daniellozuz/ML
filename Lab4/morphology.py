import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, closing, opening, disk
from skimage.measure import find_contours
from skimage import img_as_ubyte
import cv2
from scipy.ndimage.morphology import binary_hit_or_miss
from skimage.restoration.inpaint import inpaint_biharmonic
from skimage.filters import sobel, gaussian


def fingerprint():
    finger_gray = plt.imread('fingerprint.jpg')[:, :, 0]
    plt.subplot(221)
    plt.imshow(finger_gray, cmap='gray')

    finger_binary = finger_gray < 128
    plt.subplot(222)
    plt.imshow(finger_binary, cmap='gray')

    finger_skeletonized = skeletonize(finger_binary)
    plt.subplot(223)
    plt.imshow(finger_skeletonized, cmap='gray')

    plt.subplot(224)
    plt.imshow(ends(finger_skeletonized), cmap='gray')

    plt.show()


def ends(skeleton):
    structures = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]]),
        np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),
    ]
    return np.any([binary_hit_or_miss(skeleton, structure) for structure in structures], axis=0)


def get_worm(image_gray):
    return closing(image_gray < 50, selem=disk(12))


def get_backbone(image_gray):
    backbone = opening(image_gray < 140, selem=disk(3))
    backbone = closing(backbone, selem=disk(4))
    backbone = opening(backbone, selem=disk(27))
    backbone = closing(backbone, selem=disk(25))  # Backbone
    return backbone


def pancreas():
    image_gray = plt.imread('ercp_roentgen.jpg')[:, :, 0]
    image_gray.setflags(write=1)
    plt.subplot(231)
    plt.imshow(image_gray, cmap='gray')

    worm = get_worm(image_gray)
    plt.subplot(232)
    plt.imshow(worm, cmap='gray')

    backbone = get_backbone(image_gray)
    plt.subplot(233)
    plt.imshow(backbone, cmap='gray')

    image_gray[backbone] = 255
    image_gray[worm] = 255
    plt.subplot(236)
    plt.imshow(image_gray, cmap='gray')

    plt.subplot(234)
    plt.imshow(skeletonize(opening(image_gray < 128, selem=disk(1))), cmap='gray')

    plt.show()


if __name__ == '__main__':
    # fingerprint()
    pancreas()
