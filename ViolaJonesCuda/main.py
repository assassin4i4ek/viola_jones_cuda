import numpy as np
from PIL import Image

from data_provider import IntegralImageDataProvider
from object_detector import ObjectDetector
from viola_jones_classifier import ViolaJonesCascadeClassifier

if __name__ == '__main__':
    positive_dir = 'C:/Users/Admin/Desktop/AI/data/multi/face/'
    negative_dir = 'C:/Users/Admin/Desktop/AI/data/multi/non-face/'
    data_provider = IntegralImageDataProvider(positive_dir, negative_dir, 19, 19,
                                              max_positive=None, max_negative=None)

    save_file = 'C:/Users/Admin/Desktop/AI/my_classifier_3_6_12_24_48_96_192_multi'
    test_file = 'C:/Users/Admin/Desktop/AI/2.jpg'
    # test_file = 'C:/Users/Admin/Desktop/AI/lena.pgm'

    # test_positive_dir = 'C:/Users/Admin/Desktop/AI/data/all/face/'
    # test_negative_dir = 'C:/Users/Admin/Desktop/AI/data/all/non-face/'
    test_positive_dir = 'C:/Users/Admin/Desktop/AI/data/train/face/'
    test_negative_dir = 'C:/Users/Admin/Desktop/AI/data/train/non-face/'
    test_data_provider = IntegralImageDataProvider(test_positive_dir, test_negative_dir, 19, 19,
                                                   max_positive=None, max_negative=None)

    c = ViolaJonesCascadeClassifier.train_new_classifier(data_provider, [3,6,12,24,48,96,192], save_file)
    c.save(save_file)
    # c = ViolaJonesCascadeClassifier.load(save_file)
    # err = c.test(test_data_provider)
    # print(err)
    # im = Image.open(test_file)
    # im.thumbnail((50, 50))
    # ObjectDetector(c).detect_objects(im)
