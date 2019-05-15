import tensorflow as tf

from fewshot.algorithms.utils import compute_centers, pairwise_euclidian_distance, cross_entropy_with_logits
from fewshot.algorithms.special_layers import ComputeCenters, EuclideanDistanceLayer

def find_centers_and_compute_logits(inputs):
    support_features, support_classes, query_features = inputs
    support_centers = compute_centers(support_features, support_classes)
    logits = -pairwise_euclidian_distance(query_features, support_centers, reduce_mean=True)
    return logits

def build_model_for_train(backbone, input_size, num_classes, reduce_mean=True):
    support_input = tf.keras.Input(shape=input_size)
    support_classes = tf.keras.Input(shape=(num_classes,))
    query_input = tf.keras.Input(shape=input_size)

    support_features = tf.keras.layers.Flatten()(backbone(support_input))
    query_features = tf.keras.layers.Flatten()(backbone(query_input))

    centers = ComputeCenters()([support_features, support_classes])

    logits = EuclideanDistanceLayer(negative=True, reduce_mean=reduce_mean)([query_features, centers])

    return tf.keras.Model(inputs=[support_input, support_classes, query_input], outputs=logits)

def prototypical_network_trainer(
        backbone, input_size, num_classes, train_fewshot_generator, n_epochs, reduce_mean=True, metrics=["accuracy",]):
    model = build_model_for_train(
        backbone=backbone,
        input_size=input_size,
        num_classes=num_classes,
        reduce_mean=reduce_mean
    )

    model.compile(optimizer="adam", loss=lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=True), metrics=metrics)
    model.fit_generator(train_fewshot_generator, epochs=n_epochs)

    return backbone
