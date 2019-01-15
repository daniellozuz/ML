import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import watershed, remove_small_objects, binary_closing, binary_erosion, disk
from skimage.color import rgb2gray
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import slic
from skimage.filters import gaussian
from skimage.feature import blob_doh


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
        plt.subplot(7, 6, n + 1)
        plt.imshow(lungs)

        gt = ground_truth(lungs)
        plt.subplot(7, 6, n + 7)
        plt.imshow(gt)

        back = background(lungs)
        plt.subplot(7, 6, n + 13)
        plt.imshow(back)

        pink = get_pink(lungs)
        ax = plt.subplot(7, 6, n + 19)
        plt.imshow(lungs[:, :, 0])
        print('Calculating blobs')
        blobs = blob_doh(lungs[:, :, 0])
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, linewidth=2, fill=False)
            ax.add_patch(c)

        cells = np.logical_not(back)
        for blob in blobs:
            y, x, r = blob
            r = 25
            cells[int(y-r):int(y+r), int(x-r):int(x+r)] = False
        plt.subplot(7, 6, n + 25)
        cells = binary_erosion(binary_fill_holes(cells), selem=disk(10))
        plt.imshow(cells)

        markers = np.zeros_like(lungs)
        markers[back] = 1
        markers[pink] = 2
        markers[cells] = 3
        water = watershed(lungs, markers)
        plt.subplot(7, 6, n + 31)
        plt.imshow(water / 3)

        prediction = remove_small_objects(gaussian(water / 3, sigma=15)[:, :, 0] > 0.97, min_size=10000)
        plt.subplot(7, 6, n + 37)
        plt.imshow(prediction)
    plt.show()


def get_pink(image):
    pink = (207-217, 143-151, 189-196)
    new_image = np.copy(image)
    new_image[:, :, 0] -= 212
    new_image[:, :, 1] -= 148
    new_image[:, :, 2] -= 194
    return rgb2gray(new_image) < 0.1


def ground_truth(image):
    return remove_small_objects(binary_fill_holes(image[:, :, 0] == 255), 256)


def background(image):
    return binary_closing(remove_small_objects(rgb2gray(np.abs(image - 205)) < 0.08, 256))


if __name__ == '__main__':
    jcet()
