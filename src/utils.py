import tensorflow as tf

class CrossEntropyLoss(tf.losses.Loss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    
    def call(self, input_y_true, input_y_pred, mutex_y_true, mutex_y_pred):
        input_loss = tf.losses.categorical_crossentropy(input_y_true, input_y_pred)
        mutex_loss = tf.losses.categorical_crossentropy(mutex_y_true, mutex_y_pred)
        return tf.reduce_sum([input_loss, mutex_loss])

class AdaptiveLoss(tf.losses.Loss):
    def __init__(self):
        super(AdaptiveLoss, self).__init__()
        self.cross_entropy = CrossEntropyLoss()
        self.cosine_similarity = tf.losses.CosineSimilarity()
    
    def call(self, input_y_true, input_y_pred, mutex_y_true, mutex_y_pred, v_i, v_m):
        cross_entropy_loss = self.cross_entropy(input_y_true, input_y_pred, mutex_y_true, mutex_y_pred)
        cosine_loss = -1 * self.cosine_similarity(v_i, v_m)

        exp_ce = tf.exp(tf.divide(1.0, cross_entropy_loss))
        exp_cs = tf.exp(tf.divide(1.0, cosine_loss))
        
        denominator = tf.reduce_sum([exp_ce, exp_cs])

        a_1 = tf.math.divide(exp_ce, denominator)
        a_2 = tf.math.subtract(1.0, a_1)
        # loss -> 0 and a_1 -> inf so for avoid of inf a_1
        if tf.math.is_nan(a_1):
            a_1 = 1.0
            a_2 = 0.0

        multiply_1 = tf.multiply(a_1, cross_entropy_loss)
        multiply_2 = tf.multiply(a_2, cosine_loss)
        return tf.reduce_sum([multiply_1, multiply_2])
