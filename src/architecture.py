import tensorflow as tf
from attention_blocks import MutexAttentionBlock, FuseAttentionBlock
from resnet_blocks import FirstLayer, SecondLayer, ThirdLayer, FourthLayer, FifthLayer

class MutexAttentionResModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(MutexAttentionResModel, self).__init__()
        self.first_layer = FirstLayer()
        self.first_mutex = MutexAttentionBlock()
        self.first_fusion = FuseAttentionBlock(64)
        self.second_layer = SecondLayer()
        self.second_mutex = MutexAttentionBlock()
        self.second_fusion = FuseAttentionBlock(256)
        self.third_layer = ThirdLayer()
        self.third_mutex = MutexAttentionBlock()
        self.third_fusion = FuseAttentionBlock(512)
        self.fourth_layer = FourthLayer()
        self.fourth_mutex = MutexAttentionBlock()
        self.fourth_fusion = FuseAttentionBlock(1024)
        self.fifth_layer = FifthLayer()
        self.fifth_mutex = MutexAttentionBlock()
        self.fifth_fusion = FuseAttentionBlock(2048)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.last_dense = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    def call(self, x, y):
        x = self.first_layer(x)
        y = self.first_layer(y)
        f_am = self.first_mutex(x, y)
        y = self.first_fusion(f_am, y)

        x = self.second_layer(x)
        y = self.second_layer(y)
        f_am = self.second_mutex(x, y)
        y = self.second_fusion(f_am, y)

        x = self.third_layer(x)
        y = self.third_layer(y)
        f_am = self.third_mutex(x, y)
        y = self.third_fusion(f_am, y)

        x = self.fourth_layer(x)
        y = self.fourth_layer(y)
        f_am = self.fourth_mutex(x, y)
        y = self.fourth_fusion(f_am, y)

        x = self.fifth_layer(x)
        y = self.fifth_layer(y)
        f_am = self.fifth_mutex(x, y)
        y = self.fifth_fusion(f_am, y)

        v_i = tf.reshape(x, shape=(1, -1))
        v_m = tf.reshape(y, shape=(1, -1))

        x = self.global_avg_pool(x)
        y = self.global_avg_pool(y)

        input_y_pred = self.last_dense(x)
        mutex_y_pred = self.last_dense(y)
        return input_y_pred, mutex_y_pred, v_i, v_m