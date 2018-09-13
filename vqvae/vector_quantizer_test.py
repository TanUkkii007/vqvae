import tensorflow as tf
import numpy as np
from collections import namedtuple
from hypothesis import given
from hypothesis.strategies import integers, composite
from hypothesis.extra.numpy import arrays
from vqvae.vector_quantizer import VectorQuantizer


class VectorQuantizerTestArgs(namedtuple("VectorQuantizerTestArgs", ["inputs", "embedding_dim"])):
    pass


@composite
def all_args(draw, batch_size=integers(1, 3), input_dim=integers(1, 3), embedding_dim=integers(2, 4)):
    batch_size, input_dim, embedding_dim = draw(batch_size), draw(input_dim), draw(embedding_dim)
    input_shape = [embedding_dim] * input_dim
    inputs = draw(arrays(dtype=np.float32, shape=[batch_size] + input_shape, elements=integers(-1, 1)))
    return VectorQuantizerTestArgs(inputs, embedding_dim)


class VectorQuantizerTest(tf.test.TestCase):
    '''
    Reference: https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae_test.py
    '''

    @given(args=all_args(), num_embeddings=integers(2, 8))
    def test_construct(self, args, num_embeddings):
        inputs, embedding_dim = args
        vqvae = VectorQuantizer(embedding_dim=embedding_dim,
                                num_embeddings=num_embeddings,
                                commitment_cost=0.25)

        input_tensor = tf.constant(inputs)
        vq_output = vqvae(input_tensor)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            quantize, embeddings, encodings = sess.run([vq_output.quantize, vqvae.embeddings, vq_output.encodings])

            self.assertEqual(quantize.shape, inputs.shape)
            self.assertEqual(embeddings.shape, (embedding_dim, num_embeddings))

            flat_z = inputs.reshape((-1, embedding_dim))
            distances = ((flat_z ** 2).sum(axis=1, keepdims=True)
                         - 2 * np.dot(flat_z, embeddings)
                         + (embeddings ** 2).sum(axis=0, keepdims=True))
            closest_index = np.argmax(-distances, axis=1)
            self.assertAllEqual(closest_index, np.argmax(encodings, axis=1))


if __name__ == '__main__':
    tf.test.main()
