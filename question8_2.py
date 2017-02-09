from scipy import misc
import matplotlib.pyplot as plt
import numpy as np


def readImage(path):
    """
        Helper function to read the image given the path.
    """
    im = misc.imread(path)
    return im


def conv(image, kernel):
    """
        The implementation of the 2-D discrete convolution with as defined in
        lecture 8, slide 5.
        At the boundaries you should perform zero padding.
    """
    im_len, im_width = image.shape
    m, n = len(kernel), len(kernel)
    image_padded = np.pad(image, (1, 1), 'constant', constant_values=(0, 0))
    conv_image = np.zeros((im_len, im_width))
    for i in range(len(image)):
        for j in range(len(image)):
            conv_image[i, j] = np.sum(image_padded[i:i + m, j:j + n] * kernel)
    return conv_image


def min_max_rescale(image):
    s = image.shape
    ix, iy = s[0], s[1]
    maxV, minV = image.max(), image.min()
    print(minV, maxV)

    res = np.zeros((ix,iy))
    rangeV = maxV - minV
    for x in range(ix):
        for y in range(iy):
            res[x][y] = (image[x][y] - minV) / float(rangeV) * 255
    return res


image = readImage("data/clock_noise.png")
# plt.imshow(image, cmap=plt.cm.gray)

# b.)
kernel1 = 1/9 * np.ones((3, 3))
conv_image = conv(image, kernel1)
plt.imshow(min_max_rescale(conv_image), cmap=plt.cm.gray)
plt.savefig('8_2_b_blur.png')
"""
Effect of this kernel is blurring. Convolution with this kernel just takes 9 neighboring
pixels from original image, averages them and stores the result into a pixel of the new
convoluted image.
This can be useful for denoising an image.
"""

# c.)
image = readImage("data/clock.png")
kernel2 = np.array([[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]])
conv_image = conv(image, kernel2)
print(conv_image)
plt.imshow(min_max_rescale(conv_image), cmap=plt.cm.gray)
plt.savefig('8_2_c_edge_detection.png')
"""
Effect of this kernel is horizontal edge detection.
This can be useful for object segmentation or image enhancement (for example,
image sharpening).
Gray value approximately 128 means no edges in the vicinity of point.
Gray values close to 0 mean that the central pixel was white, but 2 horizontal
neighboring pixels were black.
Gray values close to 255 mean the opposite - that the central pixel was black,
but 2 horizontal neighboring pixels were white.
So both extreme black and extreme white values mean that edges were detected.

d.)
Because for calculating convolutions we use these padded zeros, which means black
pixels. The border pixels of convoluted image will be calculated using at least 3
and at most 5 black pixels.
In case of blurring the border pixels will become much darker, than neighboring pixels
that were calculated on "real" pixels.
In case of horizontal edge detection, the effect is similar. The border pixels will
become much darker and in the context of edge detection that will mean that on the borders
there will always be edges, that is not true. The mathematical concept, that this kernel
implements is gradient - how much the neighboring pixels change in horizontal direction.

A more suitable padding seems to be a padding with more natural values to the vicinity of
image than black pixels. For example, we can just replicate the border pixel or take
average of 2 border pixels. Then there will be no edges after convolution which is true.
"""
