import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import closing, square, disk, convex_hull_image
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_tv_chambolle
from skimage.filters.rank import equalize
from skimage.filters import gaussian, threshold_otsu, sobel


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


# def beautify_lungs():
#     lungs_disrupted = plt.imread('lungs.jpg').astype(np.float64)
#     print(lungs_disrupted.shape)
#     print(lungs_disrupted.dtype)
#     plt.subplot(222)
#     plt.imshow(lungs_disrupted, cmap='gray')
#     plt.show()


if __name__ == '__main__':
    beautify_head()
    # beautify_lungs()
