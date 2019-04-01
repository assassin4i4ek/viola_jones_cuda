import numpy as np
from PIL import Image

from data_provider import IntegralImageDataProvider
from integral_image import get_integral_image, integral_window
from object_detector import ObjectDetector
from viola_jones_classifier import ViolaJonesCascadeClassifier

if __name__ == '__main__':
    test_positive_dir = 'C:/Users/Admin/Desktop/AI/data/multi/face/'
    test_negative_dir = 'C:/Users/Admin/Desktop/AI/data/multi/non-face/'
    test_data_provider = IntegralImageDataProvider(test_positive_dir, test_negative_dir, 19, 19,
                                                   max_positive=0, max_negative=None)

    save_file = 'C:/Users/Admin/Desktop/AI/my_classifier_3_6_12_24_48_96_192_multi'
    test_file = 'C:/Users/Admin/Desktop/AI/2.jpg'

    # c = ViolaJonesCascadeClassifier.train_new_classifier(test_data_provider, (3,6,12,24,48,96,192), save_file)
    # c.save(save_file)
    c = ViolaJonesCascadeClassifier.load(save_file)
    # err = c.test(test_data_provider)
    # print(err)
    ObjectDetector(c).parallel_detect_objects(Image.open(test_file), 'RED')
