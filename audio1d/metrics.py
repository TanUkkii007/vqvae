import tensorflow as tf
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from librosa.display import waveplot


def plot_wav(path, y_hat, y_target, key, global_step, sample_rate):
    plt.figure(figsize=(16, 9))
    plt.subplot(2, 1, 1)
    waveplot(y_target, sr=sample_rate)
    plt.subplot(2, 1, 2)
    waveplot(y_hat, sr=sample_rate)
    plt.tight_layout()
    plt.suptitle(f"record: {key}, global step: {global_step}")
    plt.savefig(path, format="png")
    plt.close()


class MetricsSaver(tf.train.SessionRunHook):

    def __init__(self, global_step_tensor, reconstructed_audio_tensor, ground_truth_audio_tensor,
                 key_tensor, save_steps,
                 mode, hparams, writer: tf.summary.FileWriter):
        self.global_step_tensor = global_step_tensor
        self.reconstructed_audio_tensor = reconstructed_audio_tensor
        self.ground_truth_audio_tensor = ground_truth_audio_tensor
        self.key_tensor = key_tensor
        self.save_steps = save_steps
        self.mode = mode
        self.hparams = hparams
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
            global_step_value, predicted_audios, ground_truth_audios, keys = run_context.session.run(
                (self.global_step_tensor, self.reconstructed_audio_tensor,
                 self.ground_truth_audio_tensor, self.key_tensor))
            for key, predicted_audio, ground_truth_audio in zip(keys, predicted_audios,
                                                                ground_truth_audios):
                output_filename = "{}_result_step{:09d}_{}.png".format(self.mode,
                                                                       global_step_value, key.decode('utf-8'))
                output_path = os.path.join(self.writer.get_logdir(), "audio_" + output_filename)
                tf.logging.info("Saving a %s result for %d at %s", self.mode, global_step_value, output_path)
                plot_wav(output_path, predicted_audio, ground_truth_audio, key.decode('utf-8'), global_step_value,
                         self.hparams.sample_rate)
