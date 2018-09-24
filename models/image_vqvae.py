import tensorflow as tf
from vqvae.vector_quantizer import vector_quantizer_factory
from image2d.modules import Encoder, Decoder, PreNet
from image2d.metrics import MetricsSaver


class ImageVQVAEModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_prediction = mode == tf.estimator.ModeKeys.PREDICT

            x = features.images

            encoder = Encoder(num_hiddens=params.encoder_num_hiddens,
                              num_residual_layers=params.encoder_num_residual_layers,
                              num_residual_hiddens=params.encoder_num_residual_hiddens)

            decoder = Decoder(out_units=3,
                              num_hiddens=params.decoder_num_hiddens,
                              num_residual_layers=params.decoder_num_residual_layers,
                              num_residual_hiddens=params.decoder_num_residual_hiddens)

            pre_vq_conv1 = PreNet(params.embedding_dim)

            vq_vae = vector_quantizer_factory(params.vector_quantizer,
                                              embedding_dim=params.embedding_dim,
                                              num_embeddings=params.num_embeddings,
                                              commitment_cost=params.commitment_cost,
                                              sampling_count=params.sampling_count)

            z = pre_vq_conv1(encoder(x))
            vq_output = vq_vae(z)
            reconstruction = decoder(vq_output.quantize)

            reconstruction_loss = tf.losses.compute_weighted_loss(tf.squared_difference(reconstruction, x))
            loss = vq_output.loss + reconstruction_loss

            global_step = tf.train.get_global_step()
            summary_writer = tf.summary.FileWriter(model_dir)

            if is_training:
                self.add_training_stats(loss=loss,
                                        reconstruction_loss=reconstruction_loss,
                                        q_latent_loss=vq_output.q_latent_loss,
                                        commitment_loss=vq_output.commitment_loss,
                                        perplexity=vq_output.perplexity,
                                        encoding_indices=vq_output.encoding_indices,
                                        learning_rate=params.learning_rate)

                optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
                train_op = optimizer.minimize(loss, global_step=global_step)

                metrics_hook = MetricsSaver(original_image_tensor=x,
                                            reconstructed_image_tensor=reconstruction,
                                            global_step_tensor=global_step,
                                            save_steps=params.save_summary_steps,
                                            mode=mode,
                                            writer=summary_writer)

                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[metrics_hook])
            elif is_validation:
                metrics_hook = MetricsSaver(original_image_tensor=x,
                                            reconstructed_image_tensor=reconstruction,
                                            global_step_tensor=global_step,
                                            save_steps=1,
                                            mode=mode,
                                            writer=summary_writer)

                eval_metric_ops = self.get_validation_metrics(reconstruction_loss=reconstruction_loss,
                                                              q_latent_loss=vq_output.q_latent_loss,
                                                              commitment_loss=vq_output.commitment_loss,
                                                              perplexity=vq_output.perplexity)

                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  eval_metric_ops=eval_metric_ops,
                                                  evaluation_hooks=[metrics_hook])

        super(ImageVQVAEModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def add_training_stats(loss, reconstruction_loss, q_latent_loss, commitment_loss,
                           perplexity, encoding_indices, learning_rate):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("reconstruction_loss", reconstruction_loss)
        tf.summary.scalar("q_latent_loss", q_latent_loss)
        tf.summary.scalar("commitment_loss", commitment_loss)
        tf.summary.scalar("perplexity", perplexity)
        tf.summary.histogram("encoding_indices", encoding_indices)
        tf.summary.scalar("learning_rate", learning_rate)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(reconstruction_loss, q_latent_loss, commitment_loss, perplexity):
        return {
            'reconstruction_loss': tf.metrics.mean(reconstruction_loss),
            'q_latent_loss': tf.metrics.mean(q_latent_loss),
            'commitment_loss': tf.metrics.mean(commitment_loss),
            'perplexity': tf.metrics.mean(perplexity)
        }
