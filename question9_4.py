import scipy.misc
import numpy as np


def pool(data, split, f):
    result = np.empty((split, split))
    w_split, h_split = data.shape[0] // split, data.shape[1] // split
    for i in range(split):
        for j in range(split):
            result[i, j] = f(w=range(i * w_split, (i + 1) * w_split - 1),
                             h=range(j * h_split, (j + 1) * h_split - 1))
    return result


def meanPool(data, split):
    return pool(data=data, split=split, f=(lambda w, h: data[w, h].mean()))


def maxPool(data, split):
    return pool(data=data, split=split, f=(lambda w, h: data[w, h].max()))


image = scipy.misc.imread('data/clock.png')
for split in (8, 4, 2, 1):
    a = maxPool(image, split)
    argmax = np.unravel_index(a.argmax(), a.shape)
    print('Split size: {split}, maximum value: {max}, index: {argmax}'.format(
        split=split, max=a[argmax], argmax=argmax))

scipy.misc.imsave('data/clockMax.png', maxPool(image, 128))
scipy.misc.imsave('data/clockMean.png', meanPool(image, 128))
