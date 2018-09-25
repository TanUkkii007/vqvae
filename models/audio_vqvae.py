import tensorflow as tf
from vqvae.vector_quantizer import vector_quantizer_factory
from audio1d.modules import Encoder, PreNet
from audio1d.metrics import MetricsSaver
from wavenet.layers.modules import ConditionProjection, ProbabilityParameterEstimator, generate_samples
from wavenet.ops.mixture_of_logistics_distribution import discretized_mix_logistic_loss, \
    sample_from_discretized_mix_logistic


def upsample_condition(condition, times):
    batch_size = tf.shape(condition)[0]
    last_dim = tf.shape(condition)[2]
    condition = tf.expand_dims(condition, 2)
    condition = tf.tile(condition, multiples=tf.convert_to_tensor([1, 1, 1, times]))
    condition = tf.reshape(condition, shape=[batch_size, -1, last_dim])
    return condition


class MultiSpeakerVQVAEModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_prediction = mode == tf.estimator.ModeKeys.PREDICT

            x = features.waveform[:, :-1, :]
            y = features.waveform[:, 1:, :]

            encoder = Encoder(num_hiddens=params.encoder_num_hiddens,
                              num_layers=params.encoder_num_layers)

            pre_vq_conv1 = PreNet(params.embedding_dim)

            condition_projection = ConditionProjection(params.residual_channels,
                                                       params.local_condition_label_dim,
                                                       use_global_condition=params.use_global_condition,
                                                       global_condition_cardinality=params.global_condition_cardinality)

            wavenet_decoder = ProbabilityParameterEstimator(
                params.filter_width, params.residual_channels, params.dilations,
                params.skip_channels, params.out_channels,
                params.use_causal_conv_bias, params.use_filter_gate_bias, params.use_output_bias,
                params.use_skip_bias, params.use_postprocessing1_bias, params.use_postprocessing2_bias)

            vq_vae = vector_quantizer_factory(params.vector_quantizer,
                                              embedding_dim=params.embedding_dim,
                                              num_embeddings=params.num_embeddings,
                                              commitment_cost=params.commitment_cost,
                                              sampling_count=params.sampling_count)

            z = pre_vq_conv1(encoder(x))
            vq_output = vq_vae(z)

            global_condition = features.speaker_id - params.speaker_id_offset

            h = condition_projection(vq_output.quantize, global_condition=global_condition)
            h = upsample_condition(h, 2 ** params.encoder_num_layers)

            probability_params, _ = wavenet_decoder((x, h), sequential_inference_mode=False)

            wavenet_loss = discretized_mix_logistic_loss(y, probability_params, params.quantization_levels,
                                                         params.n_logistic_mix)
            reconstruction = sample_from_discretized_mix_logistic(probability_params, params.n_logistic_mix)

            reconstruction_loss = tf.losses.compute_weighted_loss(
                tf.squared_difference(reconstruction, tf.squeeze(y, axis=2)))
            loss = vq_output.loss + reconstruction_loss

            global_step = tf.train.get_global_step()

            if is_training:
                self.add_training_stats(loss=loss,
                                        reconstruction_loss=reconstruction_loss,
                                        q_latent_loss=vq_output.q_latent_loss,
                                        commitment_loss=vq_output.commitment_loss,
                                        perplexity=vq_output.perplexity,
                                        log_perplexity=vq_output.log_perplexity,
                                        encoding_indices=vq_output.encoding_indices,
                                        decoder_loss=wavenet_loss,
                                        learning_rate=params.learning_rate)

                optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
                train_op = optimizer.minimize(loss, global_step=global_step)

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
            elif is_validation:
                summary_writer = tf.summary.FileWriter(model_dir)
                metrics_saver = MetricsSaver(global_step, reconstruction, tf.squeeze(y, axis=2),
                                             features.key,
                                             1,
                                             mode, params, summary_writer)
                eval_metric_ops = self.get_validation_metrics(reconstruction_loss=reconstruction_loss,
                                                              q_latent_loss=vq_output.q_latent_loss,
                                                              commitment_loss=vq_output.commitment_loss,
                                                              perplexity=vq_output.perplexity,
                                                              log_perplexity=vq_output.log_perplexity,
                                                              decoder_loss=wavenet_loss)

                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  eval_metric_ops=eval_metric_ops,
                                                  evaluation_hooks=[metrics_saver])
            elif is_prediction:
                predicted_waveform = generate_samples(wavenet_decoder, h, params.n_logistic_mix)
                return tf.estimator.EstimatorSpec(mode, predictions={
                    "key": features.key,
                    "predicted_waveform": predicted_waveform,
                    "ground_truth_waveform": tf.squeeze(features.waveform, axis=2),
                    "speaker_id": features.speaker_id,
                })

        super(MultiSpeakerVQVAEModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def add_training_stats(loss, reconstruction_loss, q_latent_loss, commitment_loss,
                           perplexity, log_perplexity, encoding_indices, decoder_loss, learning_rate):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("reconstruction_loss", reconstruction_loss)
        tf.summary.scalar("q_latent_loss", q_latent_loss)
        tf.summary.scalar("commitment_loss", commitment_loss)
        tf.summary.scalar("perplexity", perplexity)
        tf.summary.scalar("log_perplexity", log_perplexity)
        tf.summary.histogram("encoding_indices", encoding_indices)
        tf.summary.scalar("decoder_loss", decoder_loss)
        tf.summary.scalar("learning_rate", learning_rate)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(reconstruction_loss, q_latent_loss, commitment_loss, perplexity, log_perplexity,
                               decoder_loss):
        return {
            'reconstruction_loss': tf.metrics.mean(reconstruction_loss),
            'q_latent_loss': tf.metrics.mean(q_latent_loss),
            'commitment_loss': tf.metrics.mean(commitment_loss),
            'perplexity': tf.metrics.mean(perplexity),
            'log_perplexity': tf.metrics.mean(log_perplexity),
            'decoder_loss': tf.metrics.mean(decoder_loss)
        }
