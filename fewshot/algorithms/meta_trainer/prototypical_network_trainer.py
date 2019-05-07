import tensorflow as tf

from fewshot.algorithms.utils import compute_centers, pairwise_cosine, cross_entropy_with_logits


def build_model_for_train(feature_extractor, input_size, num_classes):
    support_input = tf.keras.Input(shape=input_size)
    support_classes = tf.keras.Input(shape=(num_classes,))
    query_input = tf.keras.Input(shape=input_size)

    support_features = feature_extractor(support_input)
    query_features = feature_extractor(query_input)

    support_centers = compute_centers(support_features, query_features)

    logits = pairwise_cosine(query_features, support_centers, transpose_Y=True)

    return tf.keras.Model(inputs=[support_input, support_classes, query_input], output=logits)

def prototypical_network_trainer(
        feature_extractor, input_size, num_classes, train_fewshot_generator, n_epochs, metrics=("accuracy",)):
    model = build_model_for_train(
        feature_extractor=feature_extractor,
        input_size=input_size,
        num_classes=num_classes
    )

    model.compile(optimizer="adam", loss=cross_entropy_with_logits, metrics=metrics)
    model.fit_generator(train_fewshot_generator, epochs=n_epochs)

    return feature_extractor
