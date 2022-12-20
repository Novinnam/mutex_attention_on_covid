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

class FuseAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, units):
        super(FuseAttentionBlock, self).__init__()
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.first_dense = tf.keras.layers.Dense(units)
        self.first_batch_norm = tf.keras.layers.BatchNormalization()
        self.second_dense = tf.keras.layers.Dense(units)
        self.second_batch_norm = tf.keras.layers.BatchNormalization()
        self.m_third_dense = tf.keras.layers.Dense(units)
        self.n_third_dense = tf.keras.layers.Dense(units)

    def call(self, f_am, y):
        f_mix = tf.add(f_am, y)

        v_c1 = self.global_avg_pool(f_mix)
        v_c2 = self.global_max_pool(f_mix)

        v_c = tf.add(v_c1, v_c2)

        z = self.first_dense(v_c)
        z = self.first_batch_norm(z)
        z = tf.nn.relu(z)

        z = self.second_dense(z)
        z = self.second_batch_norm(z)
        z = tf.nn.relu(z)

        m = self.m_third_dense(z)
        n = self.n_third_dense(z)

        concated = tf.concat([m, n], axis=0)

        a_c = tf.nn.softmax(concated)

        out = tf.add(tf.multiply(a_c, f_am), 
                     tf.multiply((1 - a_c), y))
        return out
