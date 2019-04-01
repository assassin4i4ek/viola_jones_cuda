import numpy as np
from PIL import Image


def get_integral_image(img: Image):
    integral_image = np.asarray(img, dtype=np.uint32)
    integral_image = np.append(np.zeros((1, integral_image.shape[1]), dtype=np.uint32), integral_image, axis=0)
    integral_image = np.append(np.zeros((integral_image.shape[0], 1), dtype=np.uint32), integral_image, axis=1)

    for i, j in np.ndindex((integral_image.shape[0] - 1, integral_image.shape[1] - 1)):
        integral_image[i + 1, j + 1] += integral_image[i + 1, j] - integral_image[i, j] + integral_image[i, j + 1]

    return integral_image


def sum_of_rectangle(integral_image, x1, y1, x2, y2):
    return np.long(integral_image[y1, x1]) - integral_image[y1, x2] - \
           integral_image[y2, x1] + integral_image[y2, x2]


def integral_window(integral_image, x, y, width, height):
    window_image = integral_image[y: y + height + 1, x: x + width + 1]
    assert window_image.shape == (height + 1, width + 1)
    return window_image
