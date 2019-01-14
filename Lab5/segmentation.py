import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, sobel
from skimage.color import rgb2gray
from skimage.segmentation import watershed, slic
import numpy as np


def otsu():
    folder = 'segm_image/segm_image/zadanie 1'
    images = [os.path.join(folder, image) for image in os.listdir(folder)]

    for n, image in enumerate(images):
        cells = plt.imread(image, 0)
        plt.subplot(240 + n + 1)
        plt.imshow(cells, cmap='gray')

        threshold = threshold_otsu(cells)
        cells_otsu = cells <= threshold
        plt.subplot(244 + n + 1)
        plt.imshow(cells_otsu, cmap='gray')

    plt.show()


def water():
    folder = 'segm_image/segm_image/zadanie 2'
    image_names = [
        'ISIC_0000000',
        'ISIC_0000001',
        'ISIC_0000017',
        'ISIC_0000019',
    ]
    images = [os.path.join(folder, im) for im in image_names]

    for n, image in enumerate(sorted(images)):
        cells = plt.imread(image + '.jpg', 0)
        plt.subplot(4, 4, 4 * n + 1)
        plt.imshow(cells)

        cells_gray = rgb2gray(cells)
        threshold = threshold_otsu(cells_gray)
        cells_otsu = cells_gray <= threshold
        plt.subplot(4, 4, 4 * n + 2)
        plt.imshow(cells_otsu)

        markers = np.zeros_like(cells_gray)
        markers[cells_gray < threshold - 0.2] = 1
        markers[cells_gray > threshold + 0.2] = 2

        sth = watershed(sobel(cells_gray), markers)
        plt.subplot(4, 4, 4 * n + 3)
        plt.imshow(sth)

        cells = plt.imread(image + '_Segmentation.png', 0)
        plt.subplot(4, 4, 4 * n + 4)
        plt.imshow(cells)

        show_metrics(prediction=sth, labels=cells)

    plt.show()


def show_metrics(prediction, labels):
    prediction = prediction == 1
    labels = labels == 255
    true_positive = np.sum(np.logical_and(prediction, labels))
    true_negative = np.sum(np.logical_and(np.logical_not(prediction), np.logical_not(labels)))
    false_positive = np.sum(np.logical_and(prediction, np.logical_not(labels)))
    false_negative = np.sum(np.logical_and(np.logical_not(prediction), labels))
    metrics = {
        'Sensitivity': true_positive / (true_positive + false_negative),
        'Precision': true_positive / (true_negative + false_positive),
        'Dice': 2 * true_positive / (2 * true_positive + false_positive + false_negative),
        'Jaccard': true_positive / (true_positive + false_positive + false_negative),
    }
    for metric, value in metrics.items():
        print(f'{metric:<11} {value:.2f}')


def lungs():
    lungs_color = plt.imread('segm_image/segm_image/zadanie 3/Emphysema_H_and_E.jpg', 0)
    plt.subplot(121)
    plt.imshow(lungs_color)

    plt.subplot(122)
    plt.imshow(slic(lungs_color, n_segments=4, compactness=0.1, enforce_connectivity=False))

    plt.show()


def water2():
    gfp = plt.imread('segm_image/segm_image/zadanie 1/E04_01_GFP.png', 0)
    plt.subplot(221)
    plt.imshow(gfp, cmap='gray')

    markers = np.zeros_like(gfp)
    markers[gfp > 60000] = 1
    markers[gfp < 20000] = 2

    plt.subplot(222)
    plt.imshow(markers, cmap='gray')

    watered = watershed(gfp, markers, watershed_line=True)
    plt.subplot(223)
    plt.imshow(watered, cmap='gray')

    plt.show()


if __name__ == '__main__':
    # otsu()
    water()
    # lungs()
    # water2()
