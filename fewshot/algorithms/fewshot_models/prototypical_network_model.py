import tensorflow as tf

from fewshot.algorithms.special_layers import EuclideanDistanceLayer, ComputeCenters


class PrototypicalNetworkFewShotModel:
    def __init__(self, backbone):
        self.backbone = backbone

    def _compute_centers(self, X, y):
        input_classes = tf.keras.Input((y.shape[1],))
        centers = tf.keras.Model(
            inputs=[self.backbone.get_inputs()[0], input_classes],
            outputs=ComputeCenters()([tf.keras.layers.Flatten()(self.backbone.get_outputs()[0]), input_classes])
        ).predict_on_batch([X, y])
        return centers

    def _build_model_for_predictions(self, centers, reduce_mean=True):
        features = tf.keras.layers.Flatten()(self.backbone.get_outputs()[0])
        centers = tf.keras.Input(tensor=tf.constant(centers))

        distances = EuclideanDistanceLayer(negative=True, reduce_mean=reduce_mean)([features, centers])

        return tf.keras.Model(inputs=[*self.backbone.get_inputs(), centers], outputs=distances)

    def fit(self, X, y, reduce_mean=True):
        centers = self._compute_centers(X, y)
        self.model = self._build_model_for_predictions(centers, reduce_mean=reduce_mean)

    def predict(self, X, batch_size=32):
        return self.model.predict(X, batch_size=32)
