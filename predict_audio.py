"""synthesize waveform
Usage: predict.py [options]

Options:
    --data-root=<dir>               Directory contains preprocessed features.
    --checkpoint-dir=<dir>          Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>              Hyper parameters. [default: ].
    --dataset=<name>                Dataset name.
    --test-list-file=<path>         Dataset file list for test.
    --checkpoint=<path>             Restore model from checkpoint path if given.
    --output-dir=<path>             Output directory.
    -h, --help                      Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
import os
import csv
from collections import namedtuple
from audio1d.audio_util import Audio
from audio1d.metrics import plot_wav
from datasets.vctk import DatasetSource
from models.audio_vqvae import MultiSpeakerVQVAEModel
from hparams_audio import default_params, hparams_debug_string


class PredictedAudio(namedtuple("PredictedAudio",
                                ["key",
                                 "predicted_waveform",
                                 "ground_truth_waveform",
                                 "speaker_id"])):
    pass


def predict(hparams,
            model_dir, checkpoint_path, output_dir, test_files):
    audio = Audio(hparams)

    def predict_input_fn():
        dataset = DatasetSource(tf.data.TFRecordDataset(test_files), hparams).zip().group_by_batch(
            batch_size=1)
        return dataset.dataset

    estimator = MultiSpeakerVQVAEModel(hparams, model_dir)

    predictions = map(
        lambda p: PredictedAudio(p["key"], p["predicted_waveform"], p["ground_truth_waveform"], p["speaker_id"]),
        estimator.predict(predict_input_fn, checkpoint_path=checkpoint_path))

    for v in predictions:
        key = v.key.decode('utf-8')
        audio_filename = f"{key}.wav"
        audio_filepath = os.path.join(output_dir, audio_filename)
        tf.logging.info(f"Saving {audio_filepath}")
        audio.save_wav(v.predicted_waveform, audio_filepath)
        png_filename = f"{key}.png"
        png_filepath = os.path.join(output_dir, png_filename)
        tf.logging.info(f"Saving {png_filepath}")
        # ToDo: pass global step
        plot_wav(png_filepath, v.predicted_waveform, v.ground_truth_waveform, key, 0, hparams.sample_rate)


def load_file_list(filename):
    with open(filename, newline='', mode='r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            yield row[0]


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    data_root = args["--data-root"]
    output_dir = args["--output-dir"]
    dataset_name = args["--dataset"]
    assert dataset_name in ["vctk"]
    test_list_filepath = args["--test-list-file"]

    tf.logging.set_verbosity(tf.logging.INFO)

    default_params.parse(args["--hparams"])
    tf.logging.info(hparams_debug_string(default_params))

    test_list = list(load_file_list(test_list_filepath))

    test_files = [os.path.join(data_root, f"{key}.tfrecord") for key in
                  test_list]

    predict(default_params,
            checkpoint_dir,
            checkpoint_path,
            output_dir,
            test_files)


if __name__ == '__main__':
    main()
