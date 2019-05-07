import tensorflow as tf

from fewshot.algorithms.special_layers import CosineLayer
from fewshot.algorithms.utils import compute_centers


class PrototypicalNetworkFewShotModel:
    def __init__(self, backbone):
        self.feature_extractor = backbone

    def fit(self, X, y):
        input_classes = tf.keras.Input((y.shape[1],))
        centers = tf.keras.Model(
            inputs=[self.feature_extractor.get_inputs()[0], input_classes],
            outputs=compute_centers(self.feature_extractor.get_input(), input_classes)
        ).predict_on_batch([X, y])

        cosine_layer = CosineLayer(y.shape[1])
        cosine_layer.set_weights([centers])

        self.model = tf.keras.Model(
            inputs=[self.feature_extractor.get_inputs()[0]],
            outputs=cosine_layer(self.feature_extractor.get_outputs()[0])
        )


    def predict(self, X):
        return self.model.predict_on_batch(X)
