import tensorflow as tf

class CrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    
    def call(self, input_y_true, input_y_pred, mutex_y_true, mutex_y_pred):
        input_loss = tf.losses.categorical_crossentropy(input_y_true, input_y_pred)
        mutex_loss = tf.losses.categorical_crossentropy(mutex_y_true, mutex_y_pred)
        return tf.reduce_sum([input_loss, mutex_loss])