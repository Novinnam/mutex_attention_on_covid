import tensorflow as tf

class ConvolutionalBlock(tf.keras.layers.Layer):
    '''
    convolutional block split the image size 
    '''
    def __init__(self, filters, strides=2):
        super(ConvolutionalBlock, self).__init__()
        self.first_conv = tf.keras.layers.Conv2D(filters, (1, 1), strides = strides)
        self.first_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.second_conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.second_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.third_conv = tf.keras.layers.Conv2D(filters * 4, (1, 1))
        self.third_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.skip_conv = tf.keras.layers.Conv2D(filters * 4, (1, 1), strides = strides)
        self.skip_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
    
    def call(self, inputs):
        # Layer 1
        x = self.first_conv(inputs)
        x = self.first_batch_norm(x)
        x = tf.nn.relu(x)
        # Layer 2
        x = self.second_conv(x)
        x = self.second_batch_norm(x)
        x = tf.nn.relu(x)
        # Layer 3
        x = self.third_conv(x)
        x = self.third_batch_norm(x)
        # Processing Residue with conv(1,1)
        x_skip = self.skip_conv(inputs)
        x_skip = self.skip_batch_norm(x_skip)
        # Add Residue
        out = tf.add(x, x_skip)
        out = tf.nn.relu(out)
        return out

class IdentityBlock(tf.keras.layers.Layer):
    '''
    identity block doesn't change the image size 
    '''
    def __init__(self, filters):
        super(IdentityBlock, self).__init__()
        self.first_conv = tf.keras.layers.Conv2D(filters, (1, 1))
        self.first_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.second_conv = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')
        self.second_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
        self.third_conv = tf.keras.layers.Conv2D(filters * 4, (1, 1))
        self.third_batch_norm = tf.keras.layers.BatchNormalization(axis=-1)
    
    def call(self, inputs):
        # Layer 1
        x = self.first_conv(inputs)
        x = self.first_batch_norm(x)
        x = tf.nn.relu(x)
        # Layer 2
        x = self.second_conv(x)
        x = self.second_batch_norm(x)
        x = tf.nn.relu(x)
        # Layer 3
        x = self.third_conv(x)
        x = self.third_batch_norm(x)
        # Add Residue
        out = tf.add(x, inputs)
        out = tf.nn.relu(out)
        return out

class FirstLayer(tf.keras.layers.Layer):
    '''
    First block of ResNet50
    '''
    def __init__(self):
        super(FirstLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.max_pool(x)
        out = tf.nn.relu(x)
        return out

class SecondLayer(tf.keras.layers.Layer):
    '''
    Second block of ResNet50
    '''
    def __init__(self):
        super(SecondLayer, self).__init__()
        self.convolutional_block = ConvolutionalBlock(64, strides=1)
        self.first_identity = IdentityBlock(64)
        self.second_identity = IdentityBlock(64)

    def call(self, inputs):
        x = self.convolutional_block(inputs)
        x = self.first_identity(x)
        out = self.second_identity(x)
        return out

class ThirdLayer(tf.keras.layers.Layer):
    '''
    Third block of ResNet50
    '''
    def __init__(self):
        super(ThirdLayer, self).__init__()
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
    '''
    Fourth block of ResNet50
    '''
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
    '''
    Fifth block of ResNet50
    '''
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