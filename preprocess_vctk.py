# coding: utf-8
"""
Preprocess VCTK dataset
usage: preprocess_vctk.py [options] <in_dir> <out_dir>

options:
    --hparams=<parmas>       Hyper parameters. [default: ].
    -h, --help               Show help message.

"""

import os
from collections import namedtuple
from audio1d.audio_util import Audio
import tensorflow as tf
import numpy as np
import csv
from collections.abc import Iterable
from pyspark import SparkContext, RDD
from docopt import docopt
from hparams_audio import default_params


def bytes_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecord(example: tf.train.Example, filename: str):
    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(example.SerializeToString())


def write_preprocessed_data(key: str, waveform: np.ndarray, speaker_id, age, gender, filename: str):
    raw_waveform = waveform.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'key': bytes_feature([key.encode('utf-8')]),
        'waveform': bytes_feature([raw_waveform]),
        'waveform_length': int64_feature([len(waveform)]),
        'speaker_id': int64_feature([speaker_id]),
        'age': int64_feature([age]),
        'gender': int64_feature([gender]),
    }))
    write_tfrecord(example, filename)


class SpeakerInfo(namedtuple("SpeakerInfo", ["id", "age", "gender"])):
    pass


class WavRecord(namedtuple("WavRecord", ["key", "wav_path", "speaker_info"])):
    pass


class VCTK:

    def __init__(self, in_dir, out_dir, hparams, speaker_info_filename='speaker-info.txt'):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.speaker_info_filename = speaker_info_filename
        self.audio = Audio(hparams)

    def list_wav_files(self):
        def wav_files(speaker_info: SpeakerInfo):
            wav_dir = os.path.join(self.in_dir, f"wav48/p{speaker_info.id}")
            return [os.path.join(wav_dir, wav_file) for wav_file in os.listdir(wav_dir) if wav_file.endswith('.wav')]

        def wav_records(files, speaker_info):
            return [WavRecord(os.path.basename(f).strip('.wav'), f, speaker_info) for f in files]

        return sum([wav_records(wav_files(si), si) for si in self._load_speaker_info()], [])

    def process_wavs(self, rdd: RDD):
        return rdd.map(self._process_wav)

    def _load_speaker_info(self):
        with open(os.path.join(self.in_dir, self.speaker_info_filename), mode='r', encoding='utf8') as f:
            for l in f.readlines()[1:]:
                si = l.split()
                gender = 0 if si[2] == 'F' else 1
                yield SpeakerInfo(int(si[0]), int(si[1]), gender)

    def _process_wav(self, record: WavRecord):
        wav = self.audio.load_wav(record.wav_path)
        wav = self.audio.trim(wav)
        file_path = os.path.join(self.out_dir, f"{record.key}.tfrecord")
        write_preprocessed_data(record.key, wav, record.speaker_info.id, record.speaker_info.age,
                                record.speaker_info.gender, file_path)
        return record.key


if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]

    default_params.parse(args["--hparams"])

    instance = VCTK(in_dir, out_dir, default_params)

    sc = SparkContext()

    rdd = instance.process_wavs(
        sc.parallelize(instance.list_wav_files()))

    data_file_paths = rdd.collect()

    with open(os.path.join(out_dir, 'list.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for path in data_file_paths:
            writer.writerow([path])
