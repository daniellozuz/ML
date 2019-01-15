import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, gray2rgb, label2rgb
from skimage.morphology import black_tophat, disk
from skimage.restoration.inpaint import inpaint_biharmonic
from skimage.filters import gaussian, sobel
import numpy as np
from skimage.segmentation import active_contour, slic
from skimage.future import graph


def contours():
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

        inpainted = inpaint(image=mole, mask=hair)
        plt.subplot(4, 4, 4 * n + 2)
        plt.imshow(inpainted)

        snake = get_snake(inpainted)
        contour = active_contour(inpainted, snake, alpha=0.13, w_edge=1, w_line=-1)
        ax = plt.subplot(4, 4, 4 * n + 3)
        plt.imshow(inpainted)
        ax.plot(snake[:, 0], snake[:, 1])
        ax.plot(contour[:, 0], contour[:, 1])

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


def get_snake(image):
    h, w, _ = image.shape
    s = np.linspace(0, 2 * np.pi, 400)
    x = np.clip(w // 2 + w * np.cos(s), 10, w - 10)
    y = np.clip(h // 2 + h * np.sin(s), 10, h - 10)
    return np.array([x, y]).T


def rag():
    folder = 'dyst/dyst'
    image_names = [
        'Aal052',
        'Aal078',
        'Abl121',
    ]
    images = [os.path.join(folder, im) for im in image_names]

    for n, image in enumerate(sorted(images)):
        mole = plt.imread(image + '.jpg', 0)
        mole.setflags(write=1)
        border_width = 23
        mole = mole[border_width:mole.shape[0]-border_width, border_width:mole.shape[1]-border_width, :]
        plt.subplot(3, 4, 4 * n + 1)
        plt.imshow(mole)

        labels = slic(mole)
        plt.subplot(3, 4, 4 * n + 2)
        plt.imshow(labels)

        edges = sobel(rgb2gray(mole))
        edges_rgb = gray2rgb(edges)
        rag = graph.rag_boundary(labels, edges)
        ax = plt.subplot(3, 4, 4 * n + 3)
        lc = graph.show_rag(labels, rag, edges_rgb, img_cmap=None, edge_cmap='viridis', edge_width=1.2, ax=ax)
        plt.colorbar(lc, fraction=0.03)

        labels2 = graph.merge_hierarchical(labels, rag, thresh=0.04, rag_copy=False,
                                           in_place_merge=True,
                                           merge_func=merge_boundary,
                                           weight_func=weight_boundary)
        out = label2rgb(labels2, mole, kind='avg')
        plt.subplot(3, 4, 4 * n + 4)
        plt.imshow(out)

    plt.show()


def weight_boundary(graph, src, dst, n):
    default = {'weight': 0.0, 'count': 0}
    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']
    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']
    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }


def merge_boundary(graph, src, dst):
    pass


if __name__ == '__main__':
    # contours()
    rag()
