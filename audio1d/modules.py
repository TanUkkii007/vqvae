import tensorflow as tf
from functools import reduce


class PreNet(tf.layers.Layer):

    def __init__(self, out_units,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(PreNet, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                     **kwargs)

        self._conv1d = tf.layers.Conv1D(out_units,
                                        kernel_size=1,
                                        strides=1,
                                        padding='SAME')

    def call(self, inputs, **kwargs):
        return self._conv1d(inputs)


class Encoder(tf.layers.Layer):

    def __init__(self, num_hiddens, num_layers,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(Encoder, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                      **kwargs)

        self._convolutions = [tf.layers.Conv1D(filters=num_hiddens,
                                               kernel_size=4,
                                               strides=2,
                                               padding='SAME',
                                               activation=tf.nn.relu,
                                               name=f'conv1d_{i}') for i in range(num_layers)]

    def call(self, inputs, **kwargs):
        output_convs = reduce(lambda acc, l: l(acc), self._convolutions, inputs)
        return output_convs

