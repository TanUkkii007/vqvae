import tensorflow as tf
from functools import reduce


class PreNet(tf.layers.Layer):

    def __init__(self, out_units,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(PreNet, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                     **kwargs)

        self._conv2d = tf.layers.Conv2D(out_units,
                                        kernel_size=(1, 1),
                                        strides=(1, 1))

    def call(self, inputs, **kwargs):
        return self._conv2d(inputs)


class ResidualBlock(tf.layers.Layer):

    def __init__(self, out_units, residual_index, num_residual_hiddens,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(ResidualBlock, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                            **kwargs)

        self._res3x3 = tf.layers.Conv2D(filters=num_residual_hiddens,
                                        kernel_size=(3, 3),
                                        strides=(1, 1),
                                        activation=tf.nn.relu,
                                        name=f"res3x3_{residual_index}")

        self._res1x1 = tf.layers.Conv2D(filters=out_units,
                                        kernel_size=(1, 1),
                                        strides=(1, 1),
                                        activation=None,
                                        name=f"res1x1_{residual_index}")

    def call(self, inputs, **kwargs):
        h = self._res3x3(inputs)
        h = self._res1x1(h)
        h += inputs
        return tf.nn.relu(h)


class ResidualStack(tf.layers.Layer):

    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(ResidualStack, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                            **kwargs)

        self._residual_layers = [ResidualBlock(num_hiddens, i, num_residual_hiddens, name=f"residual_block_{i}") for i
                                 in range(num_residual_layers)]

    def call(self, inputs, **kwargs):
        return reduce(lambda acc, l: l(acc), self._residual_layers, inputs)


class Encoder(tf.layers.Layer):

    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(Encoder, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                      **kwargs)

        conv2d_1 = tf.layers.Conv2D(filters=num_hiddens,
                                    kernel_size=(4, 4),
                                    strides=(2, 2),
                                    activation=tf.nn.relu)

        conv2d_2 = tf.layers.Conv2D(filters=num_hiddens,
                                    kernel_size=(4, 4),
                                    strides=(2, 2),
                                    activation=tf.nn.relu)

        conv2d_3 = tf.layers.Conv2D(filters=num_hiddens,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    activation=tf.nn.relu)

        self._convolutions = [conv2d_1, conv2d_2, conv2d_3]
        self._residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)

    def call(self, inputs, **kwargs):
        output_convs = reduce(lambda acc, l: l(acc), self._convolutions, inputs)
        residual_output = self._residual_stack(output_convs)
        return residual_output


class Decoder(tf.layers.Layer):

    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(Decoder, self).__init__(trainable=trainable, name=name, dtype=dtype,
                                      **kwargs)

        conv2d_1 = tf.layers.Conv2D(filters=num_hiddens,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    activation=tf.nn.relu)

        residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)

        conv2dt_1 = tf.layers.Conv2DTranspose(filters=num_hiddens,
                                              kernel_size=(4, 4),
                                              strides=(2, 2),
                                              activation=tf.nn.relu)

        conv2dt_2 = tf.layers.Conv2DTranspose(filters=num_hiddens,
                                              kernel_size=(4, 4),
                                              strides=(2, 2),
                                              activation=None)

        self._layers = [conv2d_1, residual_stack, conv2dt_1, conv2dt_2]

    def call(self, inputs, **kwargs):
        reconstruction = reduce(lambda acc, l: acc(l), self._layers, inputs)
        return reconstruction
