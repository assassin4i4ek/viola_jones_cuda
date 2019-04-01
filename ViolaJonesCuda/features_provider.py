from abc import ABC, abstractmethod

import numpy as np

from integral_image import sum_of_rectangle


def get_all_features_for_window_size(window_width, window_height):
    feature_dtype = np.dtype([('type', np.uint8), ('x', np.uint16), ('y', np.uint16),
                              ('width', np.uint16), ('height', np.uint16)], True)
    basic_feature_sizes = ((2, 1), (1, 2), (3, 1), (1, 3), (2, 2))
    features = []

    print('(3) Processing all possible features...')

    for i, feature_size in enumerate(basic_feature_sizes):
        basic_width = feature_size[0]
        basic_height = feature_size[1]

        for width in range(basic_width, window_width + 1, basic_width):
            for height in range(basic_height, window_height + 1, basic_height):
                for x_offset in range(window_width - width + 1):
                    for y_offset in range(window_height - height + 1):
                        features.append((i + 1, x_offset, y_offset, width, height))

    print('(3) Processed {0} features!'.format(len(features)))

    return np.array(features, dtype=feature_dtype)


class AbstractFeature(ABC):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @abstractmethod
    def evaluate(self, integral_image):
        pass


"""
Haar Feature 1:
Left block minus right block
"""


class Feature1(AbstractFeature):
    def __init__(self, x=0, y=0, width=2, height=1):
        assert width % 2 == 0
        super(Feature1, self).__init__(x, y, width, height)

    def evaluate(self, integral_image):
        left_sum = sum_of_rectangle(integral_image, self.x, self.y,
                                    self.x + self.width // 2, self.y + self.height)
        right_sum = sum_of_rectangle(integral_image, self.x + self.width // 2, self.y,
                                     self.x + self.width, self.y + self.height)
        return left_sum - right_sum


"""
Haar Feature 2:
Bottom block minus upper block
"""


class Feature2(AbstractFeature):
    def __init__(self, x=0, y=0, width=1, height=2):
        assert height % 2 == 0
        super(Feature2, self).__init__(x, y, width, height)

    def evaluate(self, integral_image):
        top_sum = sum_of_rectangle(integral_image, self.x, self.y,
                                   self.x + self.width, self.y + self.height // 2)
        bottom_sum = sum_of_rectangle(integral_image, self.x, self.y + self.height // 2,
                                      self.x + self.width, self.y + self.height)
        return bottom_sum - top_sum


"""
Haar Feature 3:
Left block - middle block + right block
"""


class Feature3(AbstractFeature):
    def __init__(self, x=0, y=0, width=3, height=1):
        assert width % 3 == 0
        super(Feature3, self).__init__(x, y, width, height)

    def evaluate(self, integral_image):
        left_sum = sum_of_rectangle(integral_image, self.x, self.y,
                                    self.x + self.width // 3, self.y + self.height)
        middle_sum = sum_of_rectangle(integral_image, self.x + self.width // 3, self.y,
                                      self.x + (self.width // 3) * 2, self.y + self.height)
        right_sum = sum_of_rectangle(integral_image, self.x + (self.width // 3) * 2, self.y,
                                     self.x + self.width, self.y + self.height)
        return left_sum - middle_sum + right_sum


"""
Haar Feature 4:
Top block - middle block + bottom block
"""


class Feature4(AbstractFeature):
    def __init__(self, x=0, y=0, width=1, height=3):
        assert height % 3 == 0
        super(Feature4, self).__init__(x, y, width, height)

    def evaluate(self, integral_image):
        top_sum = sum_of_rectangle(integral_image, self.x, self.y,
                                   self.x + self.width, self.y + self.height // 3)
        middle_sum = sum_of_rectangle(integral_image, self.x, self.y + self.height // 3,
                                      self.x + self.width, self.y + (self.height // 3) * 2)
        bottom_sum = sum_of_rectangle(integral_image, self.x, self.y + (self.height // 3) * 2,
                                      self.x + self.width, self.y + self.height)
        return top_sum - middle_sum + bottom_sum


"""
Haar Feature 5:
Right top block - left top block + left bottom block - right bottom block
"""


class Feature5(AbstractFeature):
    def __init__(self, x=0, y=0, width=2, height=2):
        assert width % 2 == 0
        assert height % 2 == 0
        super(Feature5, self).__init__(x, y, width, height)

    def evaluate(self, integral_image):
        left_top_sum = sum_of_rectangle(integral_image, self.x, self.y,
                                        self.x + self.width // 2, self.y + self.height // 2)
        right_top_sum = sum_of_rectangle(integral_image, self.x + self.width // 2, self.y,
                                         self.x + self.width, self.y + self.height // 2)
        left_bottom_sum = sum_of_rectangle(integral_image, self.x, self.y + self.height // 2,
                                           self.x + self.width // 2, self.y + self.height)
        right_bottom_sum = sum_of_rectangle(integral_image, self.x + self.width // 2, self.y + self.height // 2,
                                            self.x + self.width, self.y + self.height)
        return left_top_sum - right_top_sum + right_bottom_sum - left_bottom_sum


class FeatureFactory:
    @classmethod
    def of_type(cls, feature_type, x, y, width, height):
        if feature_type == 1:
            return Feature1(x, y, width, height)
        elif feature_type == 2:
            return Feature2(x, y, width, height)
        elif feature_type == 3:
            return Feature3(x, y, width, height)
        elif feature_type == 4:
            return Feature4(x, y, width, height)
        elif feature_type == 5:
            return Feature5(x, y, width, height)
        else:
            return None
