from os import listdir

import numpy as np
from PIL import Image

from integral_image import get_integral_image


class IntegralImageDataProvider:
    def __init__(self, positive_dir, negative_dir, img_width, img_height,
                 max_positive=None, max_negative=None):
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.img_width = img_width
        self.img_height = img_height
        self.max_positive = max_positive
        self.max_negative = max_negative

    def get_integral_images_data_set(self):
        print('(2) Processing data set...')

        positive_files = listdir(self.positive_dir)[: self.max_positive]
        negative_files = listdir(self.negative_dir)[: self.max_negative]
        max_positive = len(positive_files)
        max_negative = len(negative_files)
        image_size = (self.img_width, self.img_height)

        labels = np.empty(max_positive + max_negative, dtype=np.uint8)
        images = np.empty((max_positive + max_negative, self.img_height + 1, self.img_width + 1), dtype=np.uint32)

        for i, file_name in enumerate(positive_files):
            if i % 1000 == 0 and i != 0:
                print('\t(2.1) Processed {0}/{1} positive images...'.format(i, max_positive))

            with Image.open(self.positive_dir + file_name) as img:
                if img.size != image_size:
                    # img.thumbnail(image_size)
                    img = img.resize(image_size, Image.ANTIALIAS)
                if img.mode != 'L':
                    img = img.convert('L')
                images[i] = get_integral_image(img)
                labels[i] = 1

        print('\t(2.1) Processed {} positive images!'.format(max_positive))

        for i, file_name in enumerate(negative_files):
            if i % 1000 == 0 and i != 0:
                print('\t(2.2) Processed {0}/{1} negative images...'.format(i, max_negative))

            with Image.open(self.negative_dir + file_name) as img:
                if img.size != image_size:
                    # img.thumbnail(image_size)
                    img = img.resize(image_size, Image.ANTIALIAS)
                if img.mode != 'L':
                    img = img.convert('L')
                images[max_positive + i] = get_integral_image(img)
                labels[max_positive + i] = 0

        print('\t(2.2) Processed {} negative images!'.format(max_negative))

        print('(2) Processed data set!')

        return images, labels
