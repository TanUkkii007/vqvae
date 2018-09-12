'''
Reference: https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
'''

import tensorflow as tf
from collections import namedtuple


class VectorQuantizationResult(
    namedtuple("VectorQuantizationResult", ["quantize", "perplexity", "encodings", "encoding_indices"])):
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
        z = inputs
        flat_z = tf.reshape(z, [-1, self._embedding_dim])

        distances = self.square_distance_from_e(flat_z)

        encoding_indices = tf.argmax(-distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self._num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, z.shape.as_list()[:-1])
        quantized = self.quantize(encoding_indices)

        q_latent_loss = (tf.stop_gradient(z) - quantized) ** 2
        e_latent_loss = (z - tf.stop_gradient(quantized)) ** 2
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        self.add_loss(
            tf.losses.compute_weighted_loss(loss), inputs=inputs)

        quantized = z + tf.stop_gradient(quantized - z)
        avg_probs = tf.reduce_mean(encodings, axis=0)
        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))

        return VectorQuantizationResult(quantize=quantized,
                                        perplexity=perplexity,
                                        encodings=encodings,
                                        encoding_indices=encoding_indices)

    @ @property
    def embeddings(self):
        return self._e

    def square_distance_from_e(self, z):
        return (tf.reduce_sum(z ** 2, axis=1, keep_dims=True)
                - 2 * tf.matmul(z, self._e)
                + tf.reduce_sum(self._e ** 2, axis=0, keep_dims=True))

    def quantize(self, encoding_indices):
        with tf.control_dependencies([encoding_indices]):
            e = tf.transpose(self.embeddings.read_value(), perm=[1, 0])
            return tf.nn.embedding_lookup(e, encoding_indices, validate_indices=False)
