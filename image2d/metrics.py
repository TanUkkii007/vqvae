import tensorflow as tf
import os
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_reconstruction(original_images, reconstructed_images, filename):
    figure = plt.figure(figsize=(8, 8))
    ax = figure.add_subplot(2, 1, 1)
    ax.imshow(original_images, interpolation='nearest')
    ax.set_title('original data')
    ax.set_axis_off()

    ax = figure.add_subplot(2, 1, 2)
    ax.imshow(reconstructed_images, interpolation='nearest')
    ax.set_title('reconstructed data')
    ax.set_axis_off()
    figure.savefig(filename)


# ToDo: generalize for any sizes
def convert_batch_to_image_grid(image_batch):
    batch_size, image_height, image_width, n_channels = image_batch.shape
    r = 4
    c = batch_size // r
    reshaped = (image_batch.reshape(r, c, image_height, image_width, n_channels)
                .transpose(0, 2, 1, 3, 4)
                .reshape(r * image_height, c * image_width, n_channels))
    return reshaped + 0.5


class MetricsSaver(tf.train.SessionRunHook):

    def __init__(self, original_image_tensor, reconstructed_image_tensor, global_step_tensor, save_steps,
                 mode, writer: tf.summary.FileWriter):
        self.original_image_tensor = original_image_tensor
        self.reconstructed_image_tensor = reconstructed_image_tensor
        self.global_step_tensor = global_step_tensor
        self.save_steps = save_steps
        self.mode = mode
        self.writer = writer

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({
            "global_step": self.global_step_tensor
        })

    def after_run(self,
                  run_context,
                  run_values):
        stale_global_step = run_values.results["global_step"]
        if (stale_global_step + 1) % self.save_steps == 0 or stale_global_step == 0:
            global_step_value, original_images, reconstructed_images = run_context.session.run(
                (self.global_step_tensor, self.original_image_tensor, self.reconstructed_image_tensor))
            output_filename = "{}_reconstruction_step_{:09d}_{}.png".format(self.mode, global_step_value, time.time())
            output_filepath = os.path.join(self.writer.get_logdir(), output_filename)
            tf.logging.info("Saving a %s result for %d at %s", self.mode, global_step_value, output_filepath)
            plot_reconstruction(convert_batch_to_image_grid(original_images),
                                convert_batch_to_image_grid(reconstructed_images), output_filepath)
