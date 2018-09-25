'''
Reference: https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
'''

import tensorflow as tf
import tensorflow_probability as tfp
from collections import namedtuple


class VectorQuantizationResult(
    namedtuple("VectorQuantizationResult",
               ["quantize", "perplexity", "log_perplexity", "encodings", "encoding_indices", "loss", "q_latent_loss",
                "commitment_loss"])):
    pass


class VectorQuantizer(tf.layers.Layer):

    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 trainable=True, name=None, dtype=None,
                 **kwargs):
        super(VectorQuantizer, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                              **kwargs)

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

    def build(self, input_shape):
        assert input_shape[-1].value, self._embedding_dim

        self._e = self.add_weight("e",
                                  shape=[self._embedding_dim, self._num_embeddings],
                                  initializer=tf.uniform_unit_scaling_initializer())
        self.built = True

    def call(self, inputs, **kwargs):
        z = inputs  # (B, H, W, D)
        flat_z = tf.reshape(z, [-1, self._embedding_dim])  # (B*H*W, D)

        distances = self.square_distance_from_e(flat_z)

        encoding_indices = tf.argmax(-distances, axis=1)  # (B*H*W)
        encodings = tf.one_hot(encoding_indices, self._num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, shape=tf.shape(z)[:-1])  # (B, H, W)
        quantized = self.quantize(encoding_indices)  # (B, H, W, D)

        q_latent_loss = (tf.stop_gradient(z) - quantized) ** 2
        commitment_loss = (z - tf.stop_gradient(quantized)) ** 2
        loss = tf.losses.compute_weighted_loss(q_latent_loss + self._commitment_cost * commitment_loss)

        self.add_loss(loss)

        quantized = z + tf.stop_gradient(quantized - z)
        avg_probs = tf.reduce_mean(encodings, axis=0)
        log_perplexity = - tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10))
        perplexity = tf.exp(log_perplexity)

        return VectorQuantizationResult(quantize=quantized,
                                        perplexity=perplexity,
                                        log_perplexity=log_perplexity,
                                        encodings=encodings,
                                        encoding_indices=encoding_indices,
                                        loss=loss,
                                        q_latent_loss=tf.reduce_mean(q_latent_loss),
                                        commitment_loss=tf.reduce_mean(commitment_loss))

    @property
    def embeddings(self):
        return self._e

    def square_distance_from_e(self, z):
        '''
        :param z:
        :return: (B*H*W, K)
        '''
        return (tf.reduce_sum(z ** 2, axis=1, keepdims=True)  # (B*H*W, 1)
                - 2 * tf.matmul(z, self._e)  # (B*H*W, K)
                + tf.reduce_sum(self._e ** 2, axis=0, keepdims=True))  # (1, K)

    def quantize(self, encoding_indices):
        with tf.control_dependencies([encoding_indices]):
            e = tf.transpose(self.embeddings.read_value(), perm=[1, 0])
            return tf.nn.embedding_lookup(e, encoding_indices, validate_indices=False)


class EMVectorQuantizer(tf.layers.Layer):

    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 sampling_count,
                 trainable=True, name=None, dtype=None,
                 **kwargs):
        super(EMVectorQuantizer, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                                **kwargs)

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._sampling_count = sampling_count

    def build(self, input_shape):
        assert input_shape[-1].value, self._embedding_dim

        self._e = self.add_weight("e",
                                  shape=[self._embedding_dim, self._num_embeddings],
                                  initializer=tf.uniform_unit_scaling_initializer(),
                                  trainable=False)
        self.built = True

    def call(self, inputs, **kwargs):
        z = inputs  # (B, H, W, D)
        flat_z = tf.reshape(z, [-1, self._embedding_dim])  # (B*H*W, D)

        distances = self.square_distance_from_e(flat_z)  # (B*H*W, K)

        # E step
        multinomial = tfp.distributions.Multinomial(total_count=1.0, logits=-distances)
        samples = multinomial.sample(self._sampling_count)  # (M, B*H*W, K)
        # M step
        encoder_hidden_count_per_embed = tf.reduce_sum(samples, axis=[0, 1]) / self._sampling_count  # (K)
        sample_sum = tf.reduce_sum(samples, axis=0)  # (B*H*W, K)
        new_embeddings = tf.reduce_sum(tf.expand_dims(flat_z, axis=2) * tf.expand_dims(sample_sum, axis=1),
                                       axis=0) / self._sampling_count / encoder_hidden_count_per_embed  # (D, K)
        new_embeddings = tf.assign(self.embeddings, new_embeddings)

        # sampled decoder inputs
        encoding_indices = tf.argmax(samples, axis=2)  # (M, B*H*W)
        encodings = tf.one_hot(encoding_indices, self._num_embeddings)
        encoding_indices = tf.reshape(encoding_indices,
                                      shape=tf.concat([tf.expand_dims(self._sampling_count, axis=0), tf.shape(z)[:-1]],
                                                      axis=0))  # (M, B, H, W)

        quantized = self.quantize(new_embeddings, encoding_indices)  # (B, H, W, D)

        q_latent_loss = 0
        commitment_loss = (z - tf.stop_gradient(quantized)) ** 2
        loss = tf.losses.compute_weighted_loss(q_latent_loss + self._commitment_cost * commitment_loss)

        self.add_loss(loss)

        quantized = z + tf.stop_gradient(quantized - z)
        avg_probs = tf.reduce_mean(samples, axis=[0, 1])
        log_perplexity = - tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10))
        perplexity = tf.exp(log_perplexity)

        return VectorQuantizationResult(quantize=quantized,
                                        perplexity=perplexity,
                                        log_perplexity=log_perplexity,
                                        encodings=encodings,
                                        encoding_indices=encoding_indices,
                                        loss=loss,
                                        q_latent_loss=tf.reduce_mean(q_latent_loss),
                                        commitment_loss=tf.reduce_mean(commitment_loss))

    @property
    def embeddings(self):
        return self._e

    def square_distance_from_e(self, z):
        '''
        :param z:
        :return: (B*H*W, K)
        '''
        return (tf.reduce_sum(z ** 2, axis=1, keepdims=True)  # (B*H*W, 1)
                - 2 * tf.matmul(z, self._e)  # (B*H*W, K)
                + tf.reduce_sum(self._e ** 2, axis=0, keepdims=True))  # (1, K)

    def quantize(self, embeddings, encoding_indices):
        with tf.control_dependencies([encoding_indices]):
            e = tf.transpose(embeddings, perm=[1, 0])
            sampled_embeddings = tf.nn.embedding_lookup(e, encoding_indices)  # (M, B*H*W, D)
            quantized = tf.reduce_sum(sampled_embeddings, axis=0) / self._sampling_count  # (B, H, W, D)
            return quantized


def vector_quantizer_factory(name, embedding_dim, num_embeddings, commitment_cost, sampling_count):
    if name == "VectorQuantizer":
        vq_vae = VectorQuantizer(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost)
    elif name == "EMVectorQuantizer":
        vq_vae = EMVectorQuantizer(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            sampling_count=sampling_count)
    else:
        raise ValueError(f"Unkown VectorQuantizer: {name}")
    return vq_vae
