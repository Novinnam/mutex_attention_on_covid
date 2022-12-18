import tensorflow as tf

class ConvolutionalBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvolutionalBlock, self).__init__()
        self.first_conv = tf.keras.layers.Conv2D(filters, (1, 1), strides = 2)
        self.first_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.second_conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.second_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.third_conv = tf.keras.layers.Conv2D(filters * 4, (1, 1))
        self.third_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.skip_conv = tf.keras.layers.Conv2D(filters * 4, (1, 1), strides=2)
        self.skip_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
    
    def call(self, inputs):
        x = self.first_conv(inputs)
        x = self.first_batch_norm(x)
        x = tf.nn.relu(x)

        x = self.second_conv(x)
        x = self.second_batch_norm(x)
        x = tf.nn.relu(x)

        x = self.third_conv(x)
        x = self.third_batch_norm(x)

        x_skip = self.skip_conv(inputs)
        x_skip = self.skip_batch_norm(x_skip)

        out = tf.add(x, x_skip)
        out = tf.nn.relu(out)
        return out

class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(IdentityBlock, self).__init__()
        self.first_conv = tf.keras.layers.Conv2D(filters, (1, 1))
        self.first_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.second_conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.second_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.third_conv = tf.keras.layers.Conv2D(filters * 4, (1, 1))
        self.third_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
    
    def call(self, inputs):
        x = self.first_conv(inputs)
        x = self.first_batch_norm(x)
        x = tf.nn.relu(x)

        x = self.second_conv(x)
        x = self.second_batch_norm(x)
        x = tf.nn.relu(x)

        x = self.third_conv(x)
        x = self.third_batch_norm(x)

        out = tf.add(x, inputs)
        out = tf.nn.relu(out)
        return out

class FirstLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FirstLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)
        out = self.max_pool(x)
        return out

class SecondLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SecondLayer, self).__init__()
        self.convolutional_block = ConvolutionalBlock(64)
        self.first_identity = IdentityBlock(64)
        self.second_identity = IdentityBlock(64)

    def call(self, inputs):
        x = self.convolutional_block(inputs)
        x = self.first_identity(x)
        out = self.second_identity(x)
        return out

class thirdLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(thirdLayer, self).__init__()
        self.convolutional_block = ConvolutionalBlock(128)
        self.first_identity = IdentityBlock(128)
        self.second_identity = IdentityBlock(128)
        self.third_identity = IdentityBlock(128)

    def call(self, inputs):
        x = self.convolutional_block(inputs)
        x = self.first_identity(x)
        x = self.second_identity(x)
        out = self.third_identity(x)
        return out

class FourthLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FourthLayer, self).__init__()
        self.convolutional_block = ConvolutionalBlock(256)
        self.first_identity = IdentityBlock(256)
        self.second_identity = IdentityBlock(256)
        self.third_identity = IdentityBlock(256)
        self.fourth_identity = IdentityBlock(256)
        self.fifth_identity = IdentityBlock(256)

    def call(self, inputs):
        x = self.convolutional_block(inputs)
        x = self.first_identity(x)
        x = self.second_identity(x)
        x = self.third_identity(x)
        x = self.fourth_identity(x)
        out = self.fifth_identity(x)
        return out

class FifthLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FifthLayer, self).__init__()
        self.convolutional_block = ConvolutionalBlock(512)
        self.first_identity = IdentityBlock(512)
        self.second_identity = IdentityBlock(512)

    def call(self, inputs):
        x = self.convolutional_block(inputs)
        x = self.first_identity(x)
        out = self.second_identity(x)
        return out