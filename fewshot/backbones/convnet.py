import tensorflow as tf


# swish block from  "Searching for activation functions" P. Ramachandran, B. Zoph, and Q. V. Le.
def swish1(x):
    return x * tf.keras.activations.sigmoid(x)


def create_activation(activation):
    if activation == 'swish1':
        return tf.keras.layers.Lambda(swish1)
    else:
        return tf.keras.layers.Activation(activation)


# ConvNet-N from "A Closer Look at Few-Shot Classification" Wei-Yu Chen.
# reference implementation: https://github.com/wyharveychen/CloserLookFewShot/blob/master/backbone.py
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, activation='relu', add_maxpool=True, **kwargs):
        self.conv = tf.keras.layers.Conv2D(out_channels, 3, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()

        if activation is not None:
            self.nl = create_activation(activation)
        else:
            self.nl = None

        if add_maxpool:
            self.maxpool = tf.keras.layers.MaxPool2D()
        else:
            self.maxpool = None

        self.out_channels = out_channels
        super(ConvBlock, self).__init__(**kwargs)

    def call(self, x):
        out = self.conv(x)
        out = self.bn(out)

        if self.nl is not None:
            out = self.nl(out)

        if self.maxpool is not None:
            out = self.maxpool(out)

        return out

    def set_trainable(self, trainable):
        self.conv.trainable = trainable
        self.bn.trainable = trainable

    def compute_output_shape(self, input_shape):
        out_shape = self.conv.compute_output_shape(input_shape)
        if self.maxpool is not None:
            out_shape = self.maxpool.compute_output_shape(out_shape)

        return out_shape


class ConvNet(tf.keras.layers.Layer):
    def __init__(self, input_size, outdim=64, depth=4):
        self.blocks = tf.keras.Sequential()
        for i in range(depth):
            self.blocks.add(ConvBlock(outdim, add_maxpool=(i < 4)))
        self.inputs, self.outputs = self._build_net(input_size)
        super(ConvNet, self).__init__()

    def build_model(self):
        return tf.keras.models.Model(self.inputs, self.outputs)

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def _build_net(self, input_size):
        input = tf.keras.layers.Input(shape=input_size)
        output = self.call(input)

        return [input], [output]

    def set_trainable(self, trainable):
        for layer in self.blocks.layers:
            layer.set_trainable(trainable)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for block in self.blocks:
            output_shape = block.compute_outpute_shape(output_shape)
        return output_shape

    def call(self, x):
        return self.blocks(x)
