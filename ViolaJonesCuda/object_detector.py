import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool

import numpy as np
from scipy.ndimage import label, labeled_comprehension

from integral_image import integral_window, get_integral_image
from PIL import ImageDraw, Image
from multiprocessing import Pool, Manager, cpu_count

from viola_jones_classifier import AbstractClassifier


class ObjectDetector:
    def __init__(self, classifier: AbstractClassifier):
        self.classifier = classifier

    def detect_objects(self, img, outline='GREEN'):
        original = img.copy()
        post_processed = img.copy()

        if img.mode != 'L':
            img = img.convert('L')

        x_step = 1
        y_step = 1
        scale = 1.25
        count = 0
        confidence_threshold = 3
        reduced_windows = []
        original_draw = ImageDraw.Draw(original)
        window_width = self.classifier.image_width
        window_height = self.classifier.image_height

        while window_width < img.size[0] and window_height < img.size[1]:
            window_stamps = np.zeros_like(img, np.uint16)
            integral_image = get_integral_image(img)

            for y in range(0, img.size[1] - window_height, y_step):
                for x in range(0, img.size[0] - window_width, x_step):
                    window = integral_window(integral_image, x, y, window_width, window_height)
                    if self.classifier.classify(window):
                        window_stamps[y, x] = window_width
                        original_draw.rectangle((x * original.size[0] // img.size[0],
                                                 y * original.size[1] // img.size[1],
                                                 (x + window_width) * original.size[0] // img.size[0],
                                                 (y + window_height) * original.size[1] // img.size[1]),
                                                outline=outline)

            def representative_windows(values, indexes):
                left = np.inf
                right = -1
                top = np.inf
                bottom = -1
                for index in indexes:
                    x = index % img.size[0]
                    y = index // img.size[0]
                    if x < left:
                        left = x
                    if x > right:
                        right = x
                    if y < top:
                        top = y
                    if y > bottom:
                        bottom = y

                confidence = indexes.size  # / (right - left + values[0])
                if confidence > confidence_threshold:
                    # return (left, top, right + values[0], bottom + values[0]), confidence
                    return ((left + right) // 2, (top + bottom) // 2,
                            (left + right) // 2 + values[0],
                            (top + bottom) // 2 + values[0]), confidence
                else:
                    return None

            labeled, num_of_windows = label(window_stamps, np.ones((3, 3), np.int))

            if num_of_windows != 0:
                merged_windows = list(filter(lambda w: w is not None,
                                             labeled_comprehension(window_stamps, labeled,
                                                                   np.arange(1, num_of_windows + 1),
                                                                   representative_windows, tuple, 0, True)))

                if len(merged_windows) != 0:
                    reduced_windows += [(tuple(v * original.size[0] // img.size[0] for v in x[0]),) +
                                        ((x[0][2] - x[0][0]) * original.size[0] // img.size[0], x[1])
                                        for x in merged_windows]

            count += 1
            img.thumbnail((img.size[0] / scale, img.size[1] / scale), Image.ANTIALIAS)

        original.show()

        reduced_windows.sort(key=lambda x: x[1])

        def center_of_rectangle(x1, y1, x2, y2):
            return (x1 + x2) / 2, (y1 + y2) / 2

        i = 0
        while i < len(reduced_windows):
            i_center_x, i_center_y = center_of_rectangle(*reduced_windows[i][0])
            i_confidence = reduced_windows[i][2]
            j = i + 1
            while j < len(reduced_windows):
                j_x1, j_y1, j_x2, j_y2 = reduced_windows[j][0]
                j_confidence = reduced_windows[j][2]
                if j_x1 <= i_center_x <= j_x2 and j_y1 <= i_center_y <= j_y2:
                    if i_confidence > j_confidence:
                        del reduced_windows[j]
                        j -= 1
                    else:
                        del reduced_windows[i]
                        i -= 1
                        j = len(reduced_windows)

                j += 1
            i += 1

        post_processed_draw = ImageDraw.Draw(post_processed)
        for rectangle in [x[0] for x in reduced_windows]:
            post_processed_draw.rectangle(rectangle, outline=outline)
        post_processed.show()

    def parallel_detect_objects(self, img: Image, outline='RED'):
        scale = 1.25
        original = img.copy()

        if img.mode != 'L':
            img = img.convert('L')
        scaled_images = []

        while self.classifier.image_width < img.size[0] and self.classifier.image_height < img.size[1]:
            scaled_images.append(img)
            img = img.resize((int(img.size[0] / scale), int(img.size[1] // scale)), Image.ANTIALIAS)

        with Pool(processes=cpu_count(), initializer=init_worker, initargs=(self.classifier, original.size)) as pool:
            reduced_windows = (list(itertools.chain(*pool.map(process_scaled_image, scaled_images))))

        reduced_windows.sort(key=lambda x: x[1])

        def center_of_rectangle(x1, y1, x2, y2):
            return (x1 + x2) / 2, (y1 + y2) / 2

        i = 0
        while i < len(reduced_windows):
            i_center_x, i_center_y = center_of_rectangle(*reduced_windows[i][0])
            i_confidence = reduced_windows[i][2]
            j = i + 1
            while j < len(reduced_windows):
                j_x1, j_y1, j_x2, j_y2 = reduced_windows[j][0]
                j_confidence = reduced_windows[j][2]
                if j_x1 <= i_center_x <= j_x2 and j_y1 <= i_center_y <= j_y2:
                    if i_confidence > j_confidence:
                        del reduced_windows[j]
                        j -= 1
                    else:
                        del reduced_windows[i]
                        i -= 1
                        j = len(reduced_windows)

                j += 1
            i += 1

        post_processed_draw = ImageDraw.Draw(original)
        for rectangle in [x[0] for x in reduced_windows]:
            post_processed_draw.rectangle(rectangle, outline=outline)
        original.show()


def init_worker(classifier: AbstractClassifier, original_size):
    global _classifier, window_width, window_height, _original_size
    _classifier = classifier
    window_width = classifier.image_width
    window_height = classifier.image_height
    _original_size = original_size


def process_scaled_image(scaled_image):
    confidence_threshold = 3

    window_stamps = np.zeros_like(scaled_image, np.uint16)
    integral_image = get_integral_image(scaled_image)

    for y in range(0, scaled_image.size[1] - window_height, 1):
        for x in range(0, scaled_image.size[0] - window_width, 1):
            window = integral_window(integral_image, x, y, window_width, window_height)
            if _classifier.classify(window):
                window_stamps[y, x] = window_width

    def representative_windows(values, indexes):
        left = np.inf
        right = -1
        top = np.inf
        bottom = -1
        for index in indexes:
            window_x = index % scaled_image.size[0]
            window_y = index // scaled_image.size[0]
            if window_x < left:
                left = window_x
            if window_x > right:
                right = window_x
            if window_y < top:
                top = window_y
            if window_y > bottom:
                bottom = window_y

        confidence = indexes.size  # / (right - left + values[0])
        if confidence > confidence_threshold:
            # return (left, top, right + values[0], bottom + values[0]), confidence
            return ((left + right) // 2, (top + bottom) // 2,
                    (left + right) // 2 + values[0],
                    (top + bottom) // 2 + values[0]), confidence
        else:
            return None

    labeled, num_of_windows = label(window_stamps, np.ones((3, 3), np.int))

    if num_of_windows != 0:
        merged_windows = list(filter(lambda w: w is not None,
                                     labeled_comprehension(window_stamps, labeled,
                                                           np.arange(1, num_of_windows + 1),
                                                           representative_windows, tuple, 0, True)))

        if len(merged_windows) != 0:
            return [(tuple(v * _original_size[0] // scaled_image.size[0] for v in x[0]),) +
                    ((x[0][2] - x[0][0]) * _original_size[0] // scaled_image.size[0], x[1])
                    for x in merged_windows]
    return []
