import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours
from math import atan2
from sklearn.preprocessing import normalize
from skimage.transform import resize
from scipy.misc import imresize


class Descriptor(object):
    def run(self):
        folder = 'baza/baza'
        image_names = [
            'ISIC_0000000',
            'ISIC_0000001',
            'ISIC_0000017',
            'ISIC_0000019',
        ]
        images = [os.path.join(folder, im) for im in image_names]
        plt.figure()
        for n, image in enumerate(sorted(images)):
            mole = plt.imread(image + '_Segmentation.png', 0)
            mole = self.modify(mole)
            contour = find_contours(mole, 128)
            plt.subplot(3, 4, n + 1)
            plt.imshow(mole)
            plt.plot(contour[0][:, 1], contour[0][:, 0])
            signature = shape_signature(contour, bins=30)
            plt.subplot(3, 4, n + 5)
            plt.bar(range(len(signature)), signature)
            fourier = fourier_descriptor(contour)
            plt.subplot(3, 4, n + 9)
            plt.bar(range(len(fourier)), fourier)

    def modify(self, mole):
        raise NotImplementedError


class Original(Descriptor):
    def modify(self, mole):
        return mole


class Rotated(Descriptor):
    def modify(self, mole):
        return np.rot90(mole, 3)


class Shifted(Descriptor):
    def modify(self, mole):
        return np.roll(np.roll(mole, 60, 0), 60, 1)


class Scaled(Descriptor):
    def modify(self, mole):
        return np.pad(imresize(mole, tuple(x // 2 for x in mole.shape), mode='L', interp='nearest'), 200, mode='constant', constant_values=0)


class Flipped(Descriptor):
    def modify(self, mole):
        return np.flip(mole, 0)


def shape_signature(contour, bins=30):
    x, y = contour[0][:, 1], contour[0][:, 0]
    cx, cy = np.mean(x), np.mean(y)
    d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    sth = zip(x - cx, y - cy, d)
    d = sorted(sth, key=lambda x: atan2(x[0], x[1]))
    d = [item[2] for item in d]
    i = np.argmax(d)
    d = np.concatenate((d[i:], d[:i]))
    d = interpolate(d, bins)
    return d


def interpolate(arr, bins):
    new_arr = np.zeros(bins)
    for b in range(bins):
        new_arr[b] = np.mean(arr[((b*len(arr))//bins):(((b+1)*len(arr))//bins)])
    return new_arr


def fourier_descriptor(contour, features=10):
    x, y = contour[0][:, 1], contour[0][:, 0]
    cx, cy = np.mean(x), np.mean(y)
    x, y = x - cx, y - cy
    p = x + y * 1j
    f = np.fft.fft(p)
    return [abs(x) for x in f[2:features]]


if __name__ == '__main__':
    Original().run()
    Rotated().run()
    Shifted().run()
    Scaled().run()
    Flipped().run()
    plt.show()
