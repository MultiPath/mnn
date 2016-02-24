import math
import scipy.misc
import numpy as np


def save_images(images, size, image_path, color=True):
    h, w = images.shape[1], images.shape[2]
    if color is True:
        img = np.zeros((h * size[0], w * size[1], 3))
    else:
        img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = math.floor(idx / size[1])
        if color is True:
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        else:
            img[j*h:j*h+h, i*w:i*w+w] = image
    #return scipy.misc.imsave(image_path, rescale_image(img))
    scipy.misc.toimage(rescale_image(img), cmin=0, cmax=255).save(image_path)


def save_movie_frames(images, image_path):
    for idx, image in enumerate(images):
        scipy.misc.toimage(rescale_image(image), cmin=0, cmax=255).save(image_path + str(idx).rjust(2, '0') + ".png")


def rescale_image(image):
    new_im = (image * 0.5 + 0.5) * 255
    return new_im
