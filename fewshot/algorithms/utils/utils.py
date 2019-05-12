import tensorflow as tf


def join_models(base_model, head_layer_to_apply):
    outputs = head_layer_to_apply(*base_model.get_outputs())
    return tf.keras.models.Model(base_model.get_inputs(), outputs)


def reset_weights(model):
    session = tf.keras.backend.get_session()
    print(session)
    for weight in model.trainable_weights:
        weight.initializer.run(session=session)


def compute_centers(features, targets):
    """

    :param features: NxD - N D-dimensional feature vectors
    :param targets: NxC - class for each object represented as one-hot vector
    :return: averaged feature vectors values within each class: CxD
    """
    return tf.matmul(targets, features, transpose_a=True) / tf.reduce_sum(targets, axis=-2)[:, None]


def pairwise_dot(X, Y, transpose_Y=True, normalize=False):
    if transpose_Y:
        Y = tf.transpose(Y)
    dot_product = tf.matmul(X, Y)

    if normalize:
        x_norm = tf.pow(tf.reduce_sum(X ** 2, axis=1, keepdims=True) + 1e-7, 0.5)
        y_norm = tf.pow(tf.reduce_sum(Y ** 2, axis=0, keepdims=True) + 1e-7, 0.5)
        return dot_product / x_norm / y_norm

    return dot_product


def pairwise_euclidian_distance(X, Y, reduce_mean=False):
    X = tf.expand_dims(X, axis=-2)
    Y = tf.expand_dims(Y, axis=-3)
    if reduce_mean:
        return tf.reduce_mean((X - Y) ** 2, -1)
    else:
        return tf.reduce_sum((X - Y) ** 2, -1)


def cross_entropy_with_logits(targets, logits):
    normalized_logits = tf.nn.log_softmax(logits, -1)
    return -tf.reduce_mean(tf.reduce_sum(targets * normalized_logits, -1))
