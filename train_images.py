"""Trainining script.
Usage: train_images.py [options]

Options:
    --data-root=<dir>            Directory contains data.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters. [default: ].
    --dataset=<name>             Dataset name.
    -h, --help                   Show this help message and exit
"""


from docopt import docopt
import tensorflow as tf
import logging
from datasets.cifar10 import DatasetSource, train_data_dict, valid_data_dict
from models.image_vqvae import ImageVQVAEModel
from hparams import default_params, hparams_debug_string


def train_and_evaluate(hparams, model_dir, data_dir):
    def train_input_fn():
        data_dict = train_data_dict(data_dir)
        dataset = DatasetSource(data_dict['images'], data_dict['labels'], hparams)\
            .zip()\
            .batch(hparams.batch_size)\
            .shuffle_and_repeat(hparams.shuffle_buffer_size, count=1)
        return dataset.dataset

    def eval_input_fn():
        data_dict = valid_data_dict(data_dir)
        dataset = DatasetSource(data_dict['images'], data_dict['labels'], hparams)\
            .zip()\
            .batch(hparams.batch_size)\
            .shuffle(hparams.shuffle_buffer_size)
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

    estimator = ImageVQVAEModel(hparams, model_dir, run_config)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    data_root = args["--data-root"]
    dataset_name = args["--dataset"]
    assert dataset_name in ["cifar10"]

    default_params.parse(args["--hparams"])

    log = logging.getLogger("tensorflow")
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(default_params.logfile)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info(hparams_debug_string(default_params))

    train_and_evaluate(default_params, checkpoint_dir, data_root)


if __name__ == '__main__':
    main()
