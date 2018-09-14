'''
Reference: https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
'''

import tensorflow as tf
import numpy as np
from collections import namedtuple
from six.moves import cPickle
from abc import abstractmethod
import os


def _unpickle(filename):
    with open(filename, 'rb') as fo:
        return cPickle.load(fo, encoding='latin1')


def _reshape_flattened_image_batch(flat_image_batch: np.ndarray):
    return flat_image_batch.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])  # convert from NCHW to NHWC


def _combine_batches(batch_list):
    reshaped = [_reshape_flattened_image_batch(batch['data']) for batch in batch_list]
    images = np.vstack(reshaped)
    labels = np.vstack([np.array(batch['labels']) for batch in batch_list]).reshape(-1, 1)
    return {'images': images, 'labels': labels}


def train_data_dict(data_dir):
    batches = [_unpickle(os.path.join(data_dir, 'data_batch_%d' % i)) for i in range(1, 5)]
    return _combine_batches(batches)


def valid_data_dict(data_dir):
    return _combine_batches([_unpickle(os.path.join(data_dir, 'data_batch_5'))])


def test_data_dict(data_dir):
    return _combine_batches([_unpickle(os.path.join(data_dir, 'test_batch'))])


class SourceData(namedtuple("SourceData", ["images", "labels"])):
    pass


class DatasetSource:

    def __init__(self, images, labels, hparams):
        self._dataset = tf.data.Dataset.from_tensor_slices(SourceData(
            images=self._normalize_images(images),
            labels=labels
        ))
        self._hparams = hparams

    @property
    def hparams(self):
        return self._hparams

    @staticmethod
    def _normalize_images(images):
        return tf.cast(images, tf.float32) / 255.0 - 0.5

    def zip(self):
        paired = self._dataset.map(lambda x: (x, x))
        return ZippedDataset(paired, self._hparams)


class DatasetBase:

    @abstractmethod
    def apply(self, dataset, hparams):
        raise NotImplementedError("apply")

    @property
    @abstractmethod
    def dataset(self):
        raise NotImplementedError("dataset")

    @property
    @abstractmethod
    def hparams(self):
        raise NotImplementedError("hparams")

    def shuffle(self, buffer_size):
        return self.apply(self.dataset.shuffle(buffer_size), self.hparams)

    def repeat(self, count=None):
        return self.apply(self.dataset.repeat(count), self.hparams)

    def shuffle_and_repeat(self, buffer_size, count=None):
        dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, count))
        return self.apply(dataset, self.hparams)


class ZippedDataset(DatasetBase):

    def __init__(self, dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    def apply(self, dataset, hparams):
        return ZippedDataset(dataset, hparams)

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def batch(self, batch_size):
        return BatchedDataset(self._dataset.batch(batch_size), self._hparams)


class BatchedDataset(DatasetBase):

    def __init__(self, dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    def apply(self, dataset, hparams):
        return BatchedDataset(dataset, self.hparams)

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def prefetch(self, buffer_size):
        return self.apply(self.dataset.prefetch(buffer_size), self.hparams)
