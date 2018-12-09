import tensorflow as tf
from collections import namedtuple
from abc import abstractmethod


class PreprocessedData(namedtuple("PreprocessedData",
                                  ["key", "waveform", "waveform_length",
                                   "speaker_id", "age", "gender"])):
    pass


class SourceData(namedtuple("SourceData", ["key", "waveform", "waveform_length", "speaker_id", "age", "gender"])):
    pass


def parse_preprocessed_data(proto):
    features = {
        'key': tf.FixedLenFeature((), tf.string),
        'waveform': tf.FixedLenFeature((), tf.string),
        'waveform_length': tf.FixedLenFeature((), tf.int64),
        'speaker_id': tf.FixedLenFeature((), tf.int64),
        'age': tf.FixedLenFeature((), tf.int64),
        'gender': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features


def decode_preprocessed_data(parsed):
    waveform_length = parsed['waveform_length']
    waveform = tf.decode_raw(parsed['waveform'], tf.float32)
    return PreprocessedData(
        key=parsed['key'],
        waveform=waveform,
        waveform_length=waveform_length,
        speaker_id=parsed['speaker_id'],
        age=parsed['age'],
        gender=parsed['gender']
    )


class DatasetSource:

    def __init__(self, dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    @property
    def hparams(self):
        return self._hparams

    @staticmethod
    def create_from_tfrecord_files(dataset_files, hparams, cycle_length=4,
                                   buffer_output_elements=None,
                                   prefetch_input_elements=None):
        dataset = tf.data.Dataset.from_generator(lambda: dataset_files, tf.string, tf.TensorShape([]))
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length, sloppy=False,
            buffer_output_elements=buffer_output_elements,
            prefetch_input_elements=prefetch_input_elements))
        return DatasetSource(dataset, hparams)

    def zip(self):
        paired = self._decode_data().map(lambda x: self._pad_audio(x)).map(lambda x: (x, x))
        return ZippedDataset(paired, self._hparams)

    def _decode_data(self):
        return self._dataset.map(lambda d: decode_preprocessed_data(parse_preprocessed_data(d)))

    def _pad_audio(self, d: PreprocessedData):
        compression_factor = 2 ** self.hparams.encoder_num_layers
        pad_length = compression_factor - d.waveform_length % compression_factor + 1  # +1 for future prediction
        waveform = tf.pad(tf.expand_dims(d.waveform, axis=1), paddings=tf.convert_to_tensor([[0, pad_length], [0, 0]]))
        waveform_length = d.waveform_length + pad_length
        return SourceData(
            key=d.key,
            waveform=waveform,
            waveform_length=waveform_length,
            speaker_id=d.speaker_id,
            age=d.age,
            gender=d.gender)


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

    def filter(self, predicate):
        return self.apply(self.dataset.filter(predicate), self.hparams)

    def filter_by_max_output_length(self):
        def predicate(s, t: PreprocessedData):
            max_output_length = self.hparams.max_output_length
            return tf.less_equal(t.waveform_length, max_output_length)

        return self.filter(predicate)

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

    def group_by_batch(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.hparams.batch_size
        approx_min_target_length = self.hparams.approx_min_target_length
        bucket_width = self.hparams.batch_bucket_width
        num_buckets = self.hparams.batch_num_buckets

        def key_func(unused_source, target):
            target_length = tf.minimum(target.waveform_length - approx_min_target_length, 0)
            bucket_id = target_length // bucket_width
            return tf.minimum(tf.to_int64(num_buckets), bucket_id)

        def reduce_func(unused_key, window: tf.data.Dataset):
            return window.padded_batch(batch_size, padded_shapes=(
                SourceData(
                    key=tf.TensorShape([]),
                    waveform=tf.TensorShape([None, 1]),
                    waveform_length=tf.TensorShape([]),
                    speaker_id=tf.TensorShape([]),
                    age=tf.TensorShape([]),
                    gender=tf.TensorShape([]),
                ),
                SourceData(
                    key=tf.TensorShape([]),
                    waveform=tf.TensorShape([None, 1]),
                    waveform_length=tf.TensorShape([]),
                    speaker_id=tf.TensorShape([]),
                    age=tf.TensorShape([]),
                    gender=tf.TensorShape([]),
                )), padding_values=(
                SourceData(
                    key="",
                    waveform=tf.to_float(0),
                    waveform_length=tf.to_int64(0),
                    speaker_id=tf.to_int64(0),
                    age=tf.to_int64(0),
                    gender=tf.to_int64(0),
                ),
                SourceData(
                    key="",
                    waveform=tf.to_float(0),
                    waveform_length=tf.to_int64(0),
                    speaker_id=tf.to_int64(0),
                    age=tf.to_int64(0),
                    gender=tf.to_int64(0),
                )))

        batched = self.dataset.apply(tf.contrib.data.group_by_window(key_func,
                                                                     reduce_func,
                                                                     window_size=batch_size * 5))
        return BatchedDataset(batched, self.hparams)


class BatchedDataset(DatasetBase):

    def __init__(self, dataset: tf.data.Dataset, hparams):
        self._dataset = dataset
        self._hparams = hparams

    def apply(self, dataset, hparams):
        return BatchedDataset(dataset, hparams)

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def prefetch(self, buffer_size):
        return self.apply(self.dataset.prefetch(buffer_size), self.hparams)
