import pickle
import time
from abc import ABC, abstractmethod

import numpy as np

from data_provider import IntegralImageDataProvider
from features_provider import get_all_features_for_window_size, AbstractFeature, FeatureFactory


class AbstractClassifier(ABC):
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height

    @abstractmethod
    def classify(self, integral_image):
        pass

    def test(self, integral_image_data_provider: IntegralImageDataProvider):
        images, labels = integral_image_data_provider.get_integral_images_data_set()
        error = 0
        for i in range(len(images)):
            if self.classify(images[i]) != labels[i]:
                error += 1

        return error / len(images)

    def save(self, file_name):
        with open(file_name + '.pkl', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name):
        with open(file_name + '.pkl', 'rb') as f:
            return pickle.load(f)


"""
Simple One Feature Classifier
"""


class SimpleClassifier(AbstractClassifier):
    def __init__(self, feature: AbstractFeature, polarity, threshold, image_width, image_height):
        super().__init__(image_width, image_height)
        self.feature = feature
        self.polarity = polarity
        self.threshold = threshold

    def classify(self, integral_image):
        assert integral_image.shape == (self.image_height + 1, self.image_width + 1)
        return 1 if self.feature.evaluate(integral_image) * self.polarity < self.threshold * self.polarity else 0


"""
Cascade of Viola Jones strong classifiers
"""


class ViolaJonesCascadeClassifier(AbstractClassifier):
    def __init__(self, links, image_width, image_height):
        super().__init__(image_width, image_height)
        self.links = links

    def classify(self, integral_image):
        for link in self.links:
            if link.classify(integral_image) == 0:
                return 0

        return 1

    @classmethod
    def train_new_classifier(cls, integral_image_data_provider: IntegralImageDataProvider, link_orders, save_file=None):
        links = []

        mod = cls.compile_cuda_code()

        images, labels = integral_image_data_provider.get_integral_images_data_set()

        features = get_all_features_for_window_size(integral_image_data_provider.img_width,
                                                    integral_image_data_provider.img_height)
        computed_feature_values = cls.compute_feature_values(mod, features, images,
                                                             integral_image_data_provider.img_width,
                                                             integral_image_data_provider.img_height)
        print('Training cascade classifier...')

        for link_order in link_orders:
            print('\tPreparing cascade link with order {0}...'.format(link_order))

            strong_classifier = ViolaJonesClassifier.train_new_classifier(link_order, mod, computed_feature_values,
                                                                          images, features, labels,
                                                                          integral_image_data_provider.img_width,
                                                                          integral_image_data_provider.img_height)

            print('\tPrepared cascade link with order {0}!'.format(link_order))

            links.append(strong_classifier)

            if save_file is not None:
                cls(links, integral_image_data_provider.img_width,
                    integral_image_data_provider.img_height).save(save_file)

            images_to_keep = np.full(len(images), True, np.bool)

            for i in range(len(images)):
                if labels[i] == 0 and strong_classifier.classify(images[i]) == 0:
                    images_to_keep[i] = False

            computed_feature_values = computed_feature_values\
                .reshape((len(features), len(images)))[:, images_to_keep].flatten()
            images = images[images_to_keep]
            labels = labels[images_to_keep]

            if labels[labels == 0].size == 0:
                print('No more false positive images')
                break
            else:
                print('{0} false positive images were kept...'.format(
                    images_to_keep[images_to_keep == True].size - labels[labels == 1].size))

        print('Trained cascade classifier')

        return cls(links, integral_image_data_provider.img_width, integral_image_data_provider.img_height)

    @classmethod
    def compute_feature_values(cls, mod, features, images, image_width, image_height,
                               max_buffer_size_mb=256, max_block_size=1024):
        from pycuda import driver as drv

        print('(4) Computing feature values...')

        compute_feature_values_func = mod.get_function('compute_feature_values')

        computed_feature_values = np.empty((len(features) * len(images)), dtype=np.int32)
        buffer_size = min(computed_feature_values.size,
                          max_buffer_size_mb * 1024 * 1024 // np.dtype(np.int32).itemsize)  # size in int32 elements
        block_size = min(max_block_size, buffer_size)
        grid_size = int((np.ceil(buffer_size / block_size)))

        computation_time = np.empty(int(np.ceil(computed_feature_values.size / buffer_size)))

        print('\t(4.2) Computing feature values using CUDA...'
              .format(buffer_size * np.dtype(np.int32).itemsize / (1024 * 1024), block_size))
        start = time.time()

        for buffer_offset in range(0, computed_feature_values.size, buffer_size):
            start_i = time.time()

            compute_feature_values_func(drv.Out(computed_feature_values[buffer_offset: buffer_offset + buffer_size]),
                                        drv.In(features),
                                        drv.In(images),
                                        np.uint32(image_width),
                                        np.uint32(image_height),
                                        np.uint32(len(images)),
                                        np.uint32(len(features)),
                                        np.uint32(buffer_offset),
                                        np.uint32(buffer_size),
                                        block=(block_size, 1, 1), grid=(grid_size, 1))

            end_i = time.time()

            print('\t\t\tComputed {0} feature values in {1:.2f} sec... ({2}/{3})'.format(
                buffer_size // len(images), end_i - start_i,
                min(computed_feature_values.size, (buffer_offset + buffer_size)) // len(images),
                len(features)))
            computation_time[buffer_offset // buffer_size] = end_i - start_i

        end = time.time()

        print('\t(4.2) Computed feature values using CUDA in {0:.2f} sec!'.format(end - start))
        print('\t  Average computation time = {0:.2f} microsec per feature.'
              .format((computation_time.sum() / len(features)) * 10 ** 6))
        print('\t  Max buffer size = {0} Mb, real buffer size = {1} Mb'
              .format(max_buffer_size_mb, buffer_size * np.dtype(np.int32).itemsize / (1024 * 1024)))
        print('\t  Max GPU block size = {0}, real GPU block size = {1}'.format(max_block_size, block_size))
        print('(4) Computed all feature values!')

        return computed_feature_values

    @classmethod
    def compile_cuda_code(cls):
        code = """
        #include <inttypes.h>
        #include <float.h> 
        
        typedef struct
        {
            uint8_t type;
            uint16_t x;
            uint16_t y;
            uint16_t width;
            uint16_t height;
        } Feature;
        
        typedef struct
        {
            float error;
            int8_t polarity;
            uint32_t value_index;
        } SimpleClassifier;

        __device__ inline uint32_t sum_of_rectangle(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2,
                                        uint32_t * integral_image, uint32_t image_width)
        {
            return integral_image[x1 + y1 * (image_width + 1)] - integral_image[x2 + y1 * (image_width + 1)]
                + integral_image[x2 + y2 * (image_width + 1)] - integral_image[x1 + y2 * (image_width + 1)];
        }
    
        __global__ void compute_feature_values( int32_t * computed_values, //out buffered
                                                Feature * features, //in not buffered
                                                uint32_t * integral_image, //in not buffered
                                                uint32_t image_width, //in
                                                uint32_t image_height, //in
                                                uint32_t image_count, //in
                                                uint32_t feature_count, //in
                                                uint32_t buffer_offset, //in
                                                uint32_t buffer_size //in
                                                )
        {
            uint32_t buffer_value_index = threadIdx.x + blockIdx.x * blockDim.x;
            uint32_t true_value_index = buffer_value_index + buffer_offset;
            if (buffer_value_index < buffer_size && true_value_index < image_count * feature_count)
            {
                uint32_t feature_index = true_value_index / image_count;
                uint32_t image_index = true_value_index % image_count;
                uint32_t * integral_image_ptr = &integral_image[image_index * (image_width + 1) * (image_height + 1)];
                Feature feature = features[feature_index];
                                
                if (feature.type == 1)
                {
                    uint32_t left_sum = sum_of_rectangle(
                        feature.x, feature.y,
                        feature.x + feature.width / 2, feature.y + feature.height,
                        integral_image_ptr, image_width);
                    uint32_t right_sum = sum_of_rectangle(
                        feature.x + feature.width / 2, feature.y,
                        feature.x + feature.width, feature.y + feature.height,
                        integral_image_ptr, image_width);
                    
                    computed_values[buffer_value_index] = left_sum - right_sum;
                }
                else if (feature.type == 2)
                {
                    uint32_t top_sum = sum_of_rectangle(
                        feature.x, feature.y,
                        feature.x + feature.width, feature.y + feature.height / 2,
                        integral_image_ptr, image_width);
                    uint32_t bottom_sum = sum_of_rectangle(
                        feature.x, feature.y + feature.height / 2,
                        feature.x + feature.width, feature.y + feature.height,
                        integral_image_ptr, image_width);
                    computed_values[buffer_value_index] = bottom_sum - top_sum;
                }
                else if (feature.type == 3)
                {
                    uint32_t left_sum = sum_of_rectangle(
                        feature.x, feature.y,
                        feature.x + feature.width / 3, feature.y + feature.height,
                        integral_image_ptr, image_width);
                    uint32_t middle_sum = sum_of_rectangle(
                        feature.x + feature.width / 3, feature.y,
                        feature.x + (feature.width / 3) * 2, feature.y + feature.height,
                        integral_image_ptr, image_width);
                    uint32_t right_sum = sum_of_rectangle(
                        feature.x + (feature.width / 3) * 2, feature.y,
                        feature.x + feature.width, feature.y + feature.height,
                        integral_image_ptr, image_width);
                    computed_values[buffer_value_index] = left_sum - middle_sum + right_sum;
                }
                else if (feature.type == 4)
                {
                    uint32_t top_sum = sum_of_rectangle(
                        feature.x, feature.y,
                        feature.x + feature.width, feature.y + feature.height / 3,
                        integral_image_ptr, image_width);
                    uint32_t middle_sum = sum_of_rectangle(
                        feature.x, feature.y + feature.height / 3,
                        feature.x + feature.width, feature.y + (feature.height / 3) * 2,
                        integral_image_ptr, image_width);
                    uint32_t bottom_sum = sum_of_rectangle(
                        feature.x, feature.y + (feature.height / 3) * 2,
                        feature.x + feature.width, feature.y + feature.height,
                        integral_image_ptr, image_width);
                    computed_values[buffer_value_index] = top_sum - middle_sum + bottom_sum;
                }
                else if (feature.type == 5)
                {
                    uint32_t left_top_sum = sum_of_rectangle(
                        feature.x, feature.y,
                        feature.x + feature.width / 2, feature.y + feature.height / 2,
                        integral_image_ptr, image_width);
                    uint32_t right_top_sum = sum_of_rectangle(
                        feature.x + feature.width / 2, feature.y,
                        feature.x + feature.width, feature.y + feature.height / 2,
                        integral_image_ptr, image_width);
                    uint32_t left_bottom_sum = sum_of_rectangle(
                        feature.x, feature.y + feature.height / 2,
                        feature.x + feature.width / 2, feature.y + feature.height,
                        integral_image_ptr, image_width);
                    uint32_t right_bottom_sum = sum_of_rectangle(
                        feature.x + feature.width / 2, feature.y + feature.height / 2,
                        feature.x + feature.width, feature.y + feature.height,
                        integral_image_ptr, image_width);
                    computed_values[buffer_value_index] = left_top_sum - right_top_sum + 
                                                            right_bottom_sum - left_bottom_sum;
                }       
            }
        }
            
        __global__ 
        void compute_simple_classifier_error(SimpleClassifier * classifiers, //out buffered
                                            int32_t * computed_feature_values, //in buffered
                                            uint8_t * labels, //in not buffered
                                            float * weights, //in not buffered
                                            uint32_t image_count, //in
                                            uint32_t feature_count, //in 
                                            uint32_t buffer_offset, //in 
                                            uint32_t buffer_size, //in
                                            float min_error_found_yet
                                            ) 
        {                
            uint32_t buffer_value_index = threadIdx.x + blockIdx.x * blockDim.x;
            uint32_t true_value_index = buffer_value_index + buffer_offset;
            if (buffer_value_index < buffer_size && true_value_index < image_count * feature_count)
            {
                int32_t threshold = computed_feature_values[buffer_value_index];
                float positive_polarity_error = 0;
                float negative_polarity_error = 0;
                                        
                int32_t * computed_feature_values_ptr = &computed_feature_values[(buffer_value_index / image_count)
                                                                    * image_count];
                    
                for (uint32_t i = 0; i < image_count; ++i)
                {
                    if (labels[i] == 1)
                    {
                        if (computed_feature_values_ptr[i] >= threshold)
                        {
                            positive_polarity_error += weights[i];    
                        }
                        if (computed_feature_values_ptr[i] <= threshold)
                        {
                            negative_polarity_error += weights[i];   
                        }
                    }
                    else
                    {
                        if (computed_feature_values_ptr[i] < threshold)
                        {
                            positive_polarity_error += weights[i];
                        }
                        if (computed_feature_values_ptr[i] > threshold)
                        {
                            negative_polarity_error += weights[i];
                        }
                    }    
                }
                
                if (positive_polarity_error < negative_polarity_error)
                {
                    classifiers[buffer_value_index].error = positive_polarity_error;
                    classifiers[buffer_value_index].polarity = 1;
                    classifiers[buffer_value_index].value_index = true_value_index;
                }
                else
                {
                    classifiers[buffer_value_index].error = negative_polarity_error;
                    classifiers[buffer_value_index].polarity = -1;    
                    classifiers[buffer_value_index].value_index = true_value_index;
                }
            }
        }
            
        __global__
        void get_min_error_classifier(
                                        SimpleClassifier * local_min_error_classifiers, //out not buffered
                                        SimpleClassifier * classifiers, //in buffered
                                        uint32_t classifiers_count //in
                                        )
        {
            uint32_t number_of_threads = blockDim.x;
            uint32_t thread_id = threadIdx.x;    
            uint32_t step_size = 1;

            while (number_of_threads > 0)
            {
                if (thread_id < number_of_threads)
                {   
                    int first = thread_id * step_size * 2 + blockDim.x * blockIdx.x * 2;
                    int second = first + step_size;
                    if (second < classifiers_count)
                    {
                        if (classifiers[second].error < classifiers[first].error)
                        {
                            classifiers[first].error = classifiers[second].error;
                            classifiers[first].polarity = classifiers[second].polarity;
                            classifiers[first].value_index = classifiers[second].value_index;
                        }
                    }
                }

                __syncthreads();
                        
                step_size <<= 1;
                if (number_of_threads == 1)
                {
                    number_of_threads = 0;
                }
                else 
                {
                    number_of_threads = (number_of_threads >> 1) + (number_of_threads & 0x01);
                }
            }
                        
            if (thread_id == 0)
            {
                local_min_error_classifiers[blockIdx.x].error = classifiers[blockDim.x * blockIdx.x * 2].error;
                local_min_error_classifiers[blockIdx.x].polarity = classifiers[blockDim.x * blockIdx.x * 2].polarity;
                local_min_error_classifiers[blockIdx.x].value_index = classifiers[blockDim.x * blockIdx.x * 2]
                                                                        .value_index;  
            } 
        }                    
        """

        # noinspection PyUnresolvedReferences
        from pycuda import autoinit
        from pycuda.compiler import SourceModule

        print('(1) Compiling CUDA code...')

        mod = SourceModule(code)  # , options=['--compiler-options', '-Wall'])

        print('(1) Compiled CUDA code!')

        return mod


"""
Strong Viola Jones classifier
"""


class ViolaJonesClassifier(AbstractClassifier):
    def __init__(self, classifiers, image_height, image_width):
        super().__init__(image_width, image_height)
        self.classifiers = classifiers
        self.threshold = sum([weighted_classifier[1] for weighted_classifier in classifiers]) / 2

    def classify(self, integral_image):
        score = 0

        for classifier, weight in self.classifiers:
            score += classifier.classify(integral_image) * weight

        return 1 if score >= self.threshold else 0

    @classmethod
    def train_new_classifier(cls, classifier_count, mod, computed_feature_values, images, features, labels,
                             image_width, image_height):
        classifiers = []
        max_classifier_weight = None
        image_count = len(images)
        feature_count = len(features)

        weights = cls.init_weights(labels)

        print('\t(5) Preparing classifiers...')
        for t in range(classifier_count):
            best_simple_classifier = cls.select_best_simple_classifier(mod, t, computed_feature_values, labels,
                                                                       weights, feature_count, image_count)

            best_simple_classifier, error = cls.to_simple_classifier(best_simple_classifier,
                                                                     computed_feature_values, features,
                                                                     image_count,
                                                                     image_width,
                                                                     image_height)

            if error == 0.0:
                if max_classifier_weight is not None:
                    classifiers.append((best_simple_classifier, max_classifier_weight * 2))
                else:
                    classifiers.append((best_simple_classifier, 1))
                print('\t(5.{0} Reached 0% error!'.format(t + 1))
                break

            beta = error / (1 - error)
            classifier_weight = np.log(1 / beta)

            if max_classifier_weight is None:
                max_classifier_weight = classifier_weight
            else:
                max_classifier_weight = max(max_classifier_weight, classifier_weight)

            cls.update_weights_for_next_classifier(best_simple_classifier, beta, weights, images, labels)

            classifiers.append((best_simple_classifier, classifier_weight))
        print('\t(5) Prepared {0} classifier(s)!'.format(len(classifiers)))
        return cls(classifiers, image_width, image_height)

    @classmethod
    def select_best_simple_classifier(cls, mod, t, computed_feature_values, labels, weights, feature_count,
                                      image_count,
                                      max_buffer_size_mb=256, max_block_size=1024):
        from pycuda import driver as drv
        print('\t\t(5.{0}) Selecting best simple classifier...'.format(t + 1))

        compute_simple_classifier_error_func = mod.get_function('compute_simple_classifier_error')
        get_min_error_classifier_func = mod.get_function('get_min_error_classifier')

        classifier_dtype = np.dtype([('error', np.float32), ('polarity', np.int8), ('value_index', np.uint32)],
                                    align=True)

        buffer_size = min(computed_feature_values.size,
                          max(int(np.floor(max_buffer_size_mb * 1024 * 1024 //
                                           np.dtype(np.int32).itemsize / image_count)), 1
                              ) * image_count)  # size in int32 elements
        # Error computation block/grid size
        classifier_error_computation_block_size = min(max_block_size, buffer_size)
        classifier_error_computation_grid_size = int((np.ceil(buffer_size / classifier_error_computation_block_size)))
        # Error computation memory allocation
        classifiers_gpu = drv.mem_alloc(classifier_dtype.itemsize * buffer_size)

        local_min_classifiers_size = int(np.ceil(buffer_size / (classifier_error_computation_block_size * 2)))
        # noinspection PyTypeChecker
        local_min_error_classifiers = np.full(local_min_classifiers_size,
                                              np.array([(np.inf, 0, -1)], classifier_dtype))
        # Min error classifier selection block/grid size
        min_error_classifier_block_size = classifier_error_computation_block_size
        min_error_classifier_grid_size = local_min_classifiers_size
        # Min error classifier selection memory allocation
        best_classifier = np.empty(1, classifier_dtype)
        best_classifier['error'] = np.inf

        training_computation_time = np.empty(int(np.ceil(computed_feature_values.size / buffer_size)))
        min_error_computation_time = np.empty_like(training_computation_time)

        start = time.time()

        for buffer_offset in range(0, computed_feature_values.size, buffer_size):
            start_i = time.time()

            compute_simple_classifier_error_func(classifiers_gpu,
                                                 drv.In(computed_feature_values[
                                                        buffer_offset: buffer_offset + buffer_size]),
                                                 drv.In(labels),
                                                 drv.In(weights),
                                                 np.uint32(image_count),
                                                 np.uint32(feature_count),
                                                 np.uint32(buffer_offset),
                                                 np.uint32(buffer_size),
                                                 np.float32(best_classifier['error']),
                                                 block=(classifier_error_computation_block_size, 1, 1),
                                                 grid=(classifier_error_computation_grid_size, 1))

            end_i = time.time()

            print('\t\t\tTrained {0} simple classifiers in {1:.2f} sec... ({2}/{3})'
                  .format(buffer_size // image_count, end_i - start_i,
                          min(computed_feature_values.size, (buffer_offset + buffer_size)) // image_count,
                          feature_count))
            training_computation_time[buffer_offset // buffer_size] = end_i - start_i

            start_i = time.time()

            get_min_error_classifier_func(drv.Out(local_min_error_classifiers),
                                          classifiers_gpu,
                                          np.uint32(buffer_size),
                                          block=(min_error_classifier_block_size, 1, 1),
                                          grid=(min_error_classifier_grid_size, 1))

            end_i = time.time()

            min_error_computation_time[buffer_offset // buffer_size] = end_i - start_i

            local_min_error_classifier = local_min_error_classifiers[local_min_error_classifiers['error'].argmin()]
            if best_classifier['error'] > local_min_error_classifier['error']:
                best_classifier = local_min_error_classifier.copy()

            print('\t\t\tSelected simple classifier with error = {0:.4f} in {1:.2f} milisec...'
                  .format(best_classifier['error'], (end_i - start_i) * 10 ** 3))

        end = time.time()

        print('\t\t(5.{0}) Selected min error simple classifier using CUDA in {1:.2f} milisec!'
              .format(t + 1, end - start))
        print('\t\t  Average training time = {0:.2f} sec per classifier.'
              .format((training_computation_time.sum() / feature_count) * 10 ** 3))
        print('\t\t  Average min error computation time = {0:.2f} milisec per classifier.'
              .format((min_error_computation_time.sum() / feature_count) * 10 ** 3))
        print('\t\t  Max buffer size = {0} Mb, real buffer size = {1} Mb'
              .format(max_buffer_size_mb, buffer_size * np.dtype(np.int32).itemsize / (1024 * 1024)))
        print('\t\t  Max GPU block size = {0}'.format(max_block_size))
        print('\t\t(5.{0}) Selected simple classifier with error {1:.4f}!'.format(t + 1, best_classifier['error']))

        classifiers_gpu.free()
        return best_classifier

    @classmethod
    def init_weights(cls, labels):
        positive_labels = labels == 1
        negative_labels = labels == 0
        num_of_positives = labels[positive_labels].size
        num_of_negatives = labels[negative_labels].size
        positive_init_weight = 1 / (2 * num_of_positives)
        negative_init_weight = 1 / (2 * num_of_negatives)

        weights = np.empty_like(labels, dtype=np.float32)
        weights[positive_labels] = positive_init_weight
        weights[negative_labels] = negative_init_weight
        return weights

    @classmethod
    def to_simple_classifier(cls, best_simple_classifier, computed_feature_values, features,
                             image_count, image_width, image_height):
        threshold = computed_feature_values[best_simple_classifier['value_index']]

        feature = features[best_simple_classifier['value_index'] // image_count]
        feature = FeatureFactory.of_type(feature['type'], feature['x'], feature['y'],
                                         feature['width'], feature['height'])

        polarity = best_simple_classifier['polarity']
        classifier = SimpleClassifier(feature, polarity, threshold, image_width, image_height)

        return classifier, best_simple_classifier['error']

    @classmethod
    def update_weights_for_next_classifier(cls, best_simple_classifier, beta, weights, images, labels):
        for i in range(len(images)):
            if best_simple_classifier.classify(images[i]) == labels[i]:
                weights[i] *= beta

        weights /= weights.sum()
