import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import black_tophat, disk
from skimage.restoration.inpaint import inpaint_biharmonic
from skimage.filters import gaussian
import numpy as np


def ex_1():
    folder = 'zadanie_2/zadanie 2'
    image_names = [
        'ISIC_0000000',
        'ISIC_0000001',
        'ISIC_0000017',
        'ISIC_0000019',
    ]
    images = [os.path.join(folder, im) for im in image_names]

    for n, image in enumerate(sorted(images)):
        mole = plt.imread(image + '.jpg', 0)
        mole.setflags(write=1)
        border_width = 3
        mole = mole[border_width:mole.shape[0]-border_width, border_width:mole.shape[1]-border_width, :]
        plt.subplot(4, 4, 4 * n + 1)
        plt.imshow(mole)

        hair = black_tophat(rgb2gray(mole), selem=disk(5)) > 0.03
        plt.subplot(4, 4, 4 * n + 2)
        plt.imshow(hair)

        inpainted = inpaint(image=mole, mask=hair)
        plt.subplot(4, 4, 4 * n + 3)
        plt.imshow(inpainted)

        mole_labels = plt.imread(image + '_Segmentation.png', 0)
        plt.subplot(4, 4, 4 * n + 4)
        plt.imshow(mole_labels)

    plt.show()


def inpaint(image, mask, fast=True):
    if fast:
        gauss = 255 * gaussian(image, sigma=40)
        image[mask] = gauss[mask]
        return image
    else:
        return inpaint_biharmonic(image, mask, multichannel=True)


if __name__ == '__main__':
    ex_1()
