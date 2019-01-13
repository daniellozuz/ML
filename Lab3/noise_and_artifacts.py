import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import opening, closing, square, disk, convex_hull_image
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_tv_chambolle, cycle_spin
from skimage.restoration.inpaint import inpaint_biharmonic
from skimage.filters.rank import equalize
from skimage.filters import gaussian, threshold_otsu, sobel, threshold_local
from skimage.feature import shape_index, canny
from skimage.transform import rescale


def beautify_head():
    head = plt.imread('head.png')[:, :, 0]

    plt.subplot(231)
    plt.imshow(head, cmap='gray')

    head_closed = denoise_tv_chambolle(head, weight=0.01)

    plt.subplot(232)
    plt.imshow(head_closed, cmap='gray')

    head_clahed = equalize_adapthist(head_closed, clip_limit=0.04)
 
    head_sobel = sobel(head) > 0.02

    plt.subplot(236)
    plt.imshow(head_sobel, cmap='gray')

    plt.subplot(233)
    plt.imshow(head_clahed, cmap='gray')

    mask = convex_hull_image(head_closed > 0.25)

    mask = sobel(head_closed, mask=(head_closed < 0.3))

    plt.subplot(235)
    plt.imshow(sobel(head_closed, mask=(head_closed < 0.3)), cmap='gray')

    head_eq = equalize(head_closed, selem=disk(30))

    plt.subplot(234)
    plt.imshow(head_eq, cmap='gray')

    plt.subplot(231)
    plt.imshow(head > 0.3, cmap='gray')

    plt.show()


def beautify_lungs():
    lungs = plt.imread('lungs.jpg').astype(np.float32) / 256.0

    plt.subplot(231)
    plt.imshow(lungs, cmap='gray')

    lungs_shape = shape_index(lungs)

    plt.subplot(234)
    plt.imshow(np.clip(lungs_shape, 0.2, 0.8), cmap='gray')

    lungs_denoised = denoise_tv_chambolle(lungs, weight=0.1)

    lungs_shape = shape_index(lungs_denoised)

    plt.subplot(235)
    plt.imshow(np.clip(lungs_shape, 0., 0.8), cmap='gray')

    lungs_shape_clipped = (lungs_shape > 0.3) & (lungs_shape < 0.55)

    plt.subplot(236)
    plt.imshow(lungs_shape_clipped, cmap='gray')

    plt.subplot(233)
    plt.imshow(closing(opening(lungs_shape_clipped), selem=disk(3)), cmap='gray')

    plt.subplot(232)
    plt.imshow(closing(lungs_shape_clipped, selem=disk(7)), cmap='gray')

    plt.show()


def remove_hair(image_name):
    roi_color = np.s_[200:400, 900:1100, :]
    isic = (plt.imread(image_name).astype(np.float32) / 256.0)[roi_color]

    plt.subplot(231)
    plt.imshow(isic, cmap='gray')

    isic_sobel = sobel(rgb2gray(isic))

    plt.subplot(232)
    plt.imshow(isic_sobel, cmap='gray')

    hair = closing(isic_sobel > 0.02, selem=disk(4))

    plt.subplot(233)
    plt.imshow(hair, cmap='gray')

    isic[hair] = 0

    plt.subplot(234)
    plt.imshow(isic)

    inpainted = inpaint_biharmonic(isic, hair, multichannel=True)

    plt.subplot(235)
    plt.imshow(inpainted)

    plt.show()


if __name__ == '__main__':
    # beautify_head()
    # beautify_lungs()
    remove_hair('isic_1.jpg')
