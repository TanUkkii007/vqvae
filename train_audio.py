"""Trainining script.
Usage: train_images.py [options]

Options:
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --training-list-file=<file>  Training list file
    --validation-list-file=<file> Validation list file
    --hparams=<parmas>           Hyper parameters. [default: ].
    --dataset=<name>             Dataset name.
    -h, --help                   Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
import logging
from multiprocessing import cpu_count
import csv
from datasets.vctk import DatasetSource
from models.audio_vqvae import MultiSpeakerVQVAEModel
from hparams_audio import default_params, hparams_debug_string


def train_and_evaluate(hparams, model_dir, training_list, validation_list):
    def train_input_fn():
        dataset = DatasetSource.create_from_tfrecord_files(training_list, hparams,
                                                           cycle_length=cpu_count(),
                                                           buffer_output_elements=hparams.interleave_buffer_output_elements,
                                                           prefetch_input_elements=hparams.interleave_prefetch_input_elements).zip().group_by_batch(
            hparams.batch_size).shuffle_and_repeat(buffer_size=hparams.suffle_buffer_size, count=1)
        return dataset.dataset

    def eval_input_fn():
        dataset = DatasetSource(tf.data.TFRecordDataset(validation_list), hparams).zip().group_by_batch(
            hparams.batch_size).shuffle_and_repeat(buffer_size=hparams.suffle_buffer_size, count=1)
        return dataset.dataset

    run_config = tf.estimator.RunConfig(save_summary_steps=hparams.save_summary_steps,
                                        save_checkpoints_steps=hparams.save_checkpoints_steps,
                                        keep_checkpoint_max=hparams.keep_checkpoint_max,
                                        keep_checkpoint_every_n_hours=hparams.keep_checkpoint_every_n_hours,
                                        log_step_count_steps=hparams.log_step_count_steps)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=hparams.num_evaluation_steps,
                                      throttle_secs=hparams.eval_throttle_secs,
                                      start_delay_secs=hparams.eval_start_delay_secs)

    estimator = MultiSpeakerVQVAEModel(hparams, model_dir, run_config)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def load_file_list(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            yield row[0]


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    training_list_file = args["--training-list-file"]
    validation_list_file = args["--validation-list-file"]
    dataset_name = args["--dataset"]
    assert dataset_name in ["vctk"]

    default_params.parse(args["--hparams"])

    training_list = load_file_list(training_list_file)
    validation_list = load_file_list(validation_list_file)

    log = logging.getLogger("tensorflow")
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(default_params.logfile)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info(hparams_debug_string(default_params))

    train_and_evaluate(default_params, checkpoint_dir, list(training_list), list(validation_list))


if __name__ == '__main__':
    main()
