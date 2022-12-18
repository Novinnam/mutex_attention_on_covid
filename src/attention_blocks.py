import tensorflow as tf
class MutexAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, ):
        super(MutexAttentionBlock, self).__init__()
    
    def call(self, x, y):
        z = tf.square(tf.subtract(x, y))
        z = tf.reshape(z, shape=(-1, x.shape[-1]))
        z = tf.nn.softmax(z, axis=0)
        z = tf.reshape(z, shape=(x.shape[-3], x.shape[-2], x.shape[-1]))
        out = tf.multiply(z, y)
        return out
