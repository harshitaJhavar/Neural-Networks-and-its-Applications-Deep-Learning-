from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import skimage
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel

#Creating an empty list which will store the image matrix of all the final output image matrix from filter so that we can average its value so as to generate the final image.
list_of_result_image_matrix = []

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
    	filtered = ndi.convolve(image, kernel, mode='wrap')
    	feats[k, 0] = filtered.mean()
    	feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


shrink = (slice(0, None, 3), slice(0, None, 3))
#hide = img_as_float(data.load('/home/harshita/tutorial8<2566267>/ex08/hide.png'))[shrink]
#Changing this input as the filtered weight array at line 17 can hold only 2 dimensional image input as from stackoverflow. So reading it from skimage.io.imread function.
hide = skimage.io.imread('/home/harshita/tutorial8<2566267>/ex08/hide.png',as_grey=True)[shrink]
image_name = 'hide'
image = hide

# prepare reference features
ref_feats = np.zeros((1,len(kernels), 2), dtype=np.double)
ref_feats[0,:, :] = compute_feats(hide, kernels)

print('Rotated images matched against references using Gabor filter banks:')

print('original: hide, rotated: 30deg, match result: ', end='')
feats = compute_feats(ndi.rotate(hide, angle=190, reshape=False), kernels)
print(image_name[match(feats, ref_feats)])

print('original: hide, rotated: 70deg, match result: ', end='')
feats = compute_feats(ndi.rotate(hide, angle=70, reshape=False), kernels)
print(image_name[match(feats, ref_feats)])


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    image_matrix = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    #Storing each resultant image matrix so that we can average all the resultant matrices element wise to get the final output image matrix               
    list_of_result_image_matrix.append(image_matrix)
    return(image_matrix)               

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in (-0.15,0.15):
    theta = theta / 4. * np.pi
    for frequency in (0.17,0.18,0.185,0.19,0.195):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d, frequency=%.2f, sigma=%.2f' % (theta * 180 / np.pi, frequency,sigma)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(image, kernel)]))
        
#fig, axes = plt.subplots(nrows=37, ncols=1, figsize=(5, 6))
plt.gray()

#fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

#axes[0].axis('off')

# Plot original images
#ax=axes[0]
label=image_name
img=image
#ax.imshow(img)
plt.imshow(img)
plt.show()
#ax.set_title(label, fontsize=9)
#ax.axis('off')
print(kernel_params)
#for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
for label, (kernel, powers) in zip(kernel_params, results):
    # Plot Gabor kernel
    #ax = ax_row
    #ax.imshow(np.real(kernel), interpolation='nearest')
    plt.imshow(np.real(kernel), interpolation='nearest')
    plt.show()
    #ax.set_ylabel(label, fontsize=7)
    #ax.set_xticks([])
    #ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    #ax=ax_row
    for patch in powers:
        #ax.imshow(patch, vmin=vmin, vmax=vmax)
        plt.imshow(patch, vmin=vmin, vmax=vmax)
        plt.show()
        #ax.axis('off')
#Plotting the final output image
#Each output image has dimension of 63X21 and there are total 36 images as output which is stored in list_of_result_image_matrix

final_image_matrix = np.zeros((63,21))
for i in range(0,63):
	for j in range(0,21):
		for k in range(0,10):
				final_image_matrix[i][j] += list_of_result_image_matrix[k][i][j]
		final_image_matrix[i][j] = final_image_matrix[i][j] / 10		
#The final output averaged image matrix        
plt.imshow(final_image_matrix)
plt.show()

