from tensorflow.python.ops.init_ops import Initializer,_compute_fans
from numpy.random import RandomState
import numpy as np
import scipy.io
import tensorflow as tf
import os
import random
import pathlib


class ComplexInit(Initializer):

    def __init__(self, kernel_size, input_dim,
                 weight_dim, nb_filters=None,
                 criterion='glorot', seed=None):

        # `weight_dim` is used as a parameter for sanity check
        # as we should not pass an integer as kernel_size when
        # the weight dimension is >= 2.
        # nb_filters == 0 if weights are not convolutional (matrix instead of filters)
        # then in such a case, weight_dim = 2.
        # (in case of 2D input):
        #     nb_filters == None and len(kernel_size) == 2 and_weight_dim == 2
        # conv1D: len(kernel_size) == 1 and weight_dim == 1
        # conv2D: len(kernel_size) == 2 and weight_dim == 2
        # conv3d: len(kernel_size) == 3 and weight_dim == 3

        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion
        self.seed = 1337 if seed is None else seed

    def __call__(self, shape, dtype=None, partition_info=None):

        if self.nb_filters is not None:
            kernel_shape = tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
        else:
            kernel_shape = (int(self.input_dim), self.kernel_size[-1])

        fan_in, fan_out = _compute_fans(
            tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
        )

        if self.criterion == 'glorot':
            s = 1. / (fan_in + fan_out)
        elif self.criterion == 'he':
            s = 1. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)
        rng = RandomState(self.seed)
        modulus = rng.rayleigh(scale=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        weight_real = modulus * np.cos(phase)
        weight_imag = modulus * np.sin(phase)
        weight = np.concatenate([weight_real, weight_imag], axis=-1)

        return weight


def complex_to_channels(image, name="complex2channels"):
    """Convert data from complex to channels."""
    im_real = tf.math.real(image)
    im_imag = tf.math.imag(image)
    image_out = tf.stack([im_real, im_imag], axis=-1)
    return image_out


def data_process2_old(data_path):
    """
    process data for train or test
    :param data_path: Train or test data path
    :return: labels, sparses, mask
    """
    X = scipy.io.loadmat(data_path)
    labels = X['labels'].astype(np.float32)
    labels = np.rollaxis(labels, 2, 0)
    labels = labels[..., np.newaxis]

    sparses = X['sparses'].astype(np.float32)
    sparses = np.rollaxis(sparses, 3, 0)
    try:
        mask = X['mask'].astype(np.float32)
    except KeyError:
        mask = X['masks'].astype(np.float32)
    mask = mask[:, :, np.newaxis]

    return labels, sparses, mask


def c2r(x):
    # convert a two channel complex image to one channel real image
    x_complex = tf.complex(x[..., 0], x[..., 1])
    x_mag = tf.abs(x_complex)
    x_mag = tf.expand_dims(x_mag, -1)
    return x_mag


def data_process1(data_path):
    """
    process data for train or test
    :param data_path: Train or test data path
    :return: labels, sparses, mask
    """
    case_list = os.listdir(data_path)
    random.shuffle(case_list)
    labels = []
    for case_name in case_list:
        case_path = os.path.join(data_path, case_name)
        case_data = scipy.io.loadmat(case_path)['mag']
        case_data = np.rollaxis(case_data, 2, 0)
        case_data = tf.cast(case_data, tf.float32)
        case_data = tf.expand_dims(case_data, -1)
        labels.append(case_data)

    labels = tf.concat(axis=0, values=[labels[i] for i in range(len(labels))])
    labels = tf.random.shuffle(labels)
    return labels


def data_process_T2(data_path):
    """
    process data for train or test
    :param data_path: Train or test data path
    :return: labels, sparses, mask
    """
    case_list = os.listdir(data_path)
    labels = []
    for case_name in case_list:
        case_path = os.path.join(data_path, case_name)
        case_data = scipy.io.loadmat(case_path)['mag']
        case_data = np.rollaxis(case_data, 2, 0)
        case_data = tf.cast(case_data, tf.float32)
        case_data = tf.expand_dims(case_data, -1)
        labels.append(case_data)

    labels = tf.concat(axis=0, values=[labels[i] for i in range(len(labels))])
    # labels = tf.random.shuffle(labels)
    return labels


def data_process2(data_path):
    """
    process data for train or test
    :param data_path: Train or test data path
    :return: labels, sparses, mask
    """
    case_list = os.listdir(data_path)
    random.shuffle(case_list)
    labels = []
    for case_name in case_list:
        case_path = os.path.join(data_path, case_name)
        case_data = scipy.io.loadmat(case_path)['data10']
        case_data = np.rollaxis(case_data, 2, 0)
        case_data = complex_to_channels(case_data)
        case_data = tf.cast(case_data, tf.float32)
        labels.append(case_data)

    labels = tf.concat(axis=0, values=[labels[i] for i in range(len(labels))])
    labels = tf.random.shuffle(labels)
    return labels


def data_process_test(data_path, case_indexs):
    """
    process data for train or test
    :param data_path: Train or test data path
    :return: labels, sparses, mask
    """
    case_list = sorted(os.listdir(data_path))
    labels = []
    for case_index in case_indexs:
        case_name = case_list[case_index]
        case_path = os.path.join(data_path, case_name)
        case_data = scipy.io.loadmat(case_path)['data10']
        case_data = np.rollaxis(case_data, 2, 0)
        # case_data = case_data[10:-10]
        case_data = complex_to_channels(case_data)
        case_data = tf.cast(case_data, tf.float32)
        labels.append(case_data)

    labels = tf.concat(axis=0, values=[labels[i] for i in range(len(labels))])
    return labels


def masks_process(masks_path):
    masks_list = os.listdir(masks_path)
    random.shuffle(masks_list)
    masks = []
    for mask_name in masks_list:
        mask_path = os.path.join(masks_path, mask_name)
        # mask = np.load(mask_path)
        mask = scipy.io.loadmat(mask_path)['masks']
        # mask = np.rollaxis(mask, 2, 0)
        masks.append(mask)

    masks = tf.concat(axis=0, values=[masks[i] for i in range(len(masks))])
    masks = masks[:5]
    return masks


def mask_7T_process(mask_path):
    mask = scipy.io.loadmat(mask_path)['mask']
    mask = mask[:, :, np.newaxis]
    return mask


def make_results_dir(test_results_path):
    """
    Make dir for test results save
    :param test_results_path:
    :return: ZF, CNN, GT
    """
    ZF_PATH = os.path.join(test_results_path, 'ZF')
    CNN_PATH = os.path.join(test_results_path, 'CNN')
    GT_PATH = os.path.join(test_results_path, 'GT')
    CONCAT_PATH = os.path.join(test_results_path, 'CONCAT')
    Metrics_PATH = os.path.join(test_results_path, 'metrics.mat')
    try:
        os.mkdir(ZF_PATH)
        os.mkdir(CNN_PATH)
        os.mkdir(GT_PATH)
        os.mkdir(CONCAT_PATH)
    except FileExistsError:
        pass
    return ZF_PATH, CNN_PATH, GT_PATH, CONCAT_PATH, Metrics_PATH


def Fourier2(x):
    x_complex = tf.complex(x[..., 0], x[..., 1])
    y_complex = tf.signal.fft2d(x_complex)
    return y_complex


def Fourier1(x):
    x_complex = tf.complex(x[..., 0], tf.zeros_like(x)[...,0])
    y_complex = tf.signal.fft2d(x_complex)
    return y_complex


def _undersample256(label):
    """Undersample imputed images"""
    mask = tf.random.shuffle(masks)
    mask = mask[0]

    kspace = Fourier2(label)
    uk = tf.keras.layers.Multiply()([kspace, mask])
    zf = tf.signal.ifft2d(uk)
    zf_real = tf.math.real(zf)
    zf_imag = tf.math.imag(zf)
    input = tf.stack([zf_real, zf_imag], axis=-1)
    return input, label, mask


def _undersample256_1(label):
    """Undersample imputed images"""
    mask = tf.random.shuffle(masks)
    mask = mask[0]

    kspace = Fourier1(label)
    uk = tf.keras.layers.Multiply()([kspace, mask])
    zf = tf.signal.ifft2d(uk)
    zf_real = tf.math.real(zf)
    zf_imag = tf.math.imag(zf)
    input = tf.stack([zf_real, zf_imag], axis=-1)
    return input, label, mask


def _undersample256_1cha(label):
    """Undersample imputed images"""
    mask = tf.random.shuffle(masks)
    mask = mask[0]

    kspace = Fourier1(label)
    uk = tf.keras.layers.Multiply()([kspace, mask])
    zf = tf.signal.ifft2d(uk)
    input = tf.expand_dims(tf.abs(zf), -1)
    label = label
    return input, label, mask


def load_7T_SWI(mat_file):
    # mat_file = mat_file.numpy()

    mat_file = mat_file.numpy()
    label = scipy.io.loadmat(mat_file)['img_full']
    input = scipy.io.loadmat(mat_file)['img_zero']

    label_real = tf.math.real(label)
    label_imag = tf.math.imag(label)

    input_real = tf.math.real(input)
    input_imag = tf.math.imag(input)

    input = tf.stack([input_real, input_imag], axis=-1)
    label = tf.stack([label_real, label_imag], axis=-1)
    return input, label


def _undersample_7T(mat_file):

    mat_file = mat_file.numpy()
    label = scipy.io.loadmat(mat_file)['img_full']
    label = complex_to_channels(label)

    # mask = tf.random.shuffle(masks)
    mask = masks[0]

    kspace = Fourier2(label)
    uk = tf.keras.layers.Multiply()([kspace, mask])
    zf = tf.signal.ifft2d(uk)
    zf_real = tf.math.real(zf)
    zf_imag = tf.math.imag(zf)
    input = tf.stack([zf_real, zf_imag], axis=-1)

    return input, label, mask


def warp_precess(path):
    [x, y] = tf.py_function(load_7T_SWI, inp=[path], Tout=[tf.float32, tf.float32])
    return x, y


def warp_precess_online(path):
    [x, y, m] = tf.py_function(_undersample_7T, inp=[path], Tout=[tf.float32, tf.float32, tf.complex64])
    return x, y, m


def data_loader(traindata_path, masks_path, buffer_size, batch_size=4):
    global masks
    labels = data_process2(traindata_path)

    masks = masks_process(masks_path)
    masks = np.fft.ifftshift(masks)
    masks = tf.cast(masks, tf.complex64)

    # Construct training data using Tensorflow data.Dataset
    data_size = labels.shape[0]
    validation_size = np.round(0.1 * data_size).astype(int)

    train_dataset = tf.data.Dataset.from_tensor_slices(labels[validation_size:-1])
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.map(_undersample256,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Construct testing data using Tensorflow data.Dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(labels[:validation_size])
    test_dataset = test_dataset.map(_undersample256)
    test_dataset = test_dataset.batch(batch_size)

    print('label_train shape/min/max: ', tf.shape(labels), tf.reduce_min(labels), tf.reduce_max(labels))
    return train_dataset, test_dataset


def data_loader_7T(traindata_path, buffer_size, batch_size=4):
    data_dir = pathlib.Path(traindata_path)
    all_image_paths = list(data_dir.glob('*.mat'))
    all_image_paths = [str(path) for path in all_image_paths]

    dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)
    dataset = dataset.map(warp_precess, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    val_batches = tf.data.experimental.cardinality(dataset)

    test_dataset = dataset.take(val_batches // 10)
    train_dataset = dataset.skip(val_batches // 10)

    # print('label_train shape: ', tf.shape(dataset))
    return train_dataset, test_dataset


def data_loader_7T_online(traindata_path, masks_path, buffer_size, batch_size=4):

    global masks
    masks = masks_process(masks_path)
    masks = np.fft.ifftshift(masks)
    masks = tf.cast(masks, tf.complex64)

    data_dir = pathlib.Path(traindata_path)
    all_image_paths = list(data_dir.glob('*.mat'))
    all_image_paths = [str(path) for path in all_image_paths]

    dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)
    dataset = dataset.map(warp_precess_online, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    val_batches = tf.data.experimental.cardinality(dataset)

    test_dataset = dataset.take(val_batches // 10)
    train_dataset = dataset.skip(val_batches // 10)

    # print('label_train shape: ', tf.shape(dataset))
    return train_dataset, test_dataset


def data_loader_7T_test(traindata_path, masks_path, batch_size=4):
    global masks
    masks = masks_process(masks_path)
    masks = np.fft.ifftshift(masks)
    masks = tf.cast(masks, tf.complex64)

    data_dir = pathlib.Path(traindata_path)
    all_image_paths = list(data_dir.glob('*.mat'))
    all_image_paths = [str(path) for path in all_image_paths]

    dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)
    dataset = dataset.map(warp_precess_online, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def data_loader_test_7T(traindata_path, batch_size=4):
    data_dir = pathlib.Path(traindata_path)
    all_image_paths = list(data_dir.glob('*.mat'))
    all_image_paths = [str(path) for path in all_image_paths]

    dataset = tf.data.Dataset.from_tensor_slices(all_image_paths)
    dataset = dataset.map(warp_precess, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # print('label_train shape: ', tf.shape(dataset))
    return dataset


def data_loader_1cha(traindata_path, masks_path, buffer_size, batch_size=4):
    global masks
    labels = data_process1(traindata_path)

    masks = masks_process(masks_path)
    masks = np.fft.ifftshift(masks)
    masks = tf.cast(masks, tf.complex64)

    # Construct training data using Tensorflow data.Dataset
    data_size = labels.shape[0]
    validation_size = np.round(0.1 * data_size).astype(int)

    train_dataset = tf.data.Dataset.from_tensor_slices(labels[validation_size:-1])
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.map(_undersample256_1,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Construct testing data using Tensorflow data.Dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(labels[:validation_size])
    test_dataset = test_dataset.map(_undersample256_1)
    test_dataset = test_dataset.batch(batch_size)

    print('label_train shape/min/max: ', tf.shape(labels), tf.reduce_min(labels), tf.reduce_max(labels))
    return train_dataset, test_dataset


def data_loader_test(traindata_path, masks_path, case_indexs, batch_size=4):
    global masks
    labels = data_process_test(traindata_path, case_indexs)
    data_size = tf.shape(labels)[0]
    masks = masks_process(masks_path)
    masks = np.fft.ifftshift(masks)
    masks = tf.cast(masks, tf.complex64)

    # Construct testing data using Tensorflow data.Dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(labels)
    test_dataset = test_dataset.map(_undersample256)
    test_dataset = test_dataset.batch(batch_size)

    print('label_train shape/min/max: ', tf.shape(labels), tf.reduce_min(labels), tf.reduce_max(labels))
    return test_dataset, data_size


def data_loader_test_T2(traindata_path, masks_path, batch_size=4):
    global masks
    labels = data_process_T2(traindata_path)
    data_size = tf.shape(labels)[0]
    masks = masks_process(masks_path)
    masks = np.fft.ifftshift(masks)
    masks = tf.cast(masks, tf.complex64)

    # Construct testing data using Tensorflow data.Dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(labels)
    test_dataset = test_dataset.map(_undersample256_1)
    test_dataset = test_dataset.batch(batch_size)

    print('label_train shape/min/max: ', tf.shape(labels), tf.reduce_min(labels), tf.reduce_max(labels))
    return test_dataset, data_size