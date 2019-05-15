import tensorflow as tf

from fewshot.algorithms.utils import pairwise_dot, pairwise_euclidian_distance, compute_centers


def pairwise_cosine(X, Y, transpose_Y=True):
    if transpose_Y:
        Y = tf.transpose(Y)

    dot_product = tf.matmul(X, Y)
    x_norm = tf.pow(tf.reduce_sum(tf.multiply(X, X), axis=1, keepdims=True) + 1e-7, 0.5)
    y_norm = tf.pow(tf.reduce_sum(tf.multiply(Y, Y), axis=0, keepdims=True) + 1e-7, 0.5)

    return dot_product / x_norm / y_norm


class CosineLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(CosineLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 2, input_shape

        self.W = self.add_weight(
            name='kernel',
            shape=(input_shape[1].value, self.num_classes),
            initializer='uniform',
            trainable=True
        )

        super(CosineLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0].value, self.num_classes)

    def call(self, input, **kwargs):
        return pairwise_dot(input, self.W, transpose_Y=True, normalize=True)


class EuclideanDistanceLayer(tf.keras.layers.Layer):
    def __init__(self, negative=False, reduce_mean=True):
        self.sign = -1 if negative else 1
        self.reduce_mean = reduce_mean
        super(EuclideanDistanceLayer, self).__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0].value, input_shape[1][0].value)

    def call(self, input):
        return self.sign * pairwise_euclidian_distance(*input, reduce_mean=self.reduce_mean)


class ComputeCenters(tf.keras.layers.Layer):
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0].value, input_shape[1][1])

    def call(self, input):
        features, targets = input
        return compute_centers(features, targets)