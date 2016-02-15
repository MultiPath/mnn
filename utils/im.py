import math
import scipy.misc
import numpy as np


def save_images(images, size, image_path):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = math.floor(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return scipy.misc.imsave(image_path, img)
