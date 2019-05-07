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

    return tf.matmul(targets, features, transpose_a=True) / tf.reduce_sum(targets, axis=0)

def pairwise_cosine(X, Y, transpose_Y=True):
    if transpose_Y:
        Y = tf.transpose(Y)

    dot_product = tf.matmul(X, Y)
    x_norm = tf.pow(tf.reduce_sum(tf.multiply(X, X), axis=1, keepdims=True), 0.5)
    y_norm = tf.pow(tf.reduce_sum(tf.multiply(Y, Y), axis=0, keepdims=True), 0.5)

    return dot_product / x_norm / y_norm

def cross_entropy_with_logits(targets, logits):
    normalized_logits = tf.nn.log_softmax(logits, -1)
    return tf.reduce_mean(tf.reduce_sum(targets * normalized_logits, -1))
