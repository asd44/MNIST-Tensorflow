import tensorflow as tf
from .mlp_model import MLPFactory


class CNNFactory:
    @staticmethod
    def pool_2x2(x):
        return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    @staticmethod
    def conv_layer(x, shape, activation=tf.nn.relu):
        conv_w = MLPFactory.normal_var(shape)
        conv_b = MLPFactory.normal_var([shape[3]])
        conv = tf.nn.conv2d(x, conv_w, [1, 1, 1, 1], 'SAME')
        return activation(tf.nn.bias_add(conv, conv_b))

    @staticmethod
    def conv_noramalized_layer(x, shape, activation=tf.nn.relu):
        conv_w = MLPFactory.normal_var(shape)
        conv = tf.nn.conv2d(x, conv_w, [1, 1, 1, 1], 'SAME')
        bn = MLPFactory.batch_normalizer(conv)
        return activation(bn)

    @staticmethod
    def conv_pool_localnorm_layer(x, shape, activation=tf.nn.relu):
        conv = CNNFactory.conv_layer(x, shape, activation=activation)
        pool = CNNFactory.pool_2x2(conv)
        return tf.nn.local_response_normalization(pool)

    @staticmethod
    def conv_pool_batchnorm_layer(x, shape, activation=tf.nn.relu):
        conv = CNNFactory.conv_noramalized_layer(x, shape, activation=activation)
        return CNNFactory.pool_2x2(conv)

    @staticmethod
    def create_cnn(input, layers, activation=tf.nn.relu, batch_normalize=False):
        current_layers = input.get_shape().as_list()[3]
        current_tensor = input

        if batch_normalize:
            cpnl = CNNFactory.conv_pool_batchnorm_layer
        else:
            cpnl = CNNFactory.conv_pool_localnorm_layer

        for params in layers:
            current_tensor = cpnl(current_tensor, [params[0], params[0], current_layers, params[1]],
                                  activation=activation)
            current_layers = params[1]

        return current_tensor


class CNN:
    def __init__(self, shape, x_dtype, y_dtype):
        self.__shape = shape
        self.__ph_x = tf.placeholder(x_dtype, shape=[None, shape[0][0], shape[0][1], shape[0][2]])
        self.__ph_y = tf.placeholder(y_dtype, shape=[None, shape[1]])
        self.__keepprob = tf.placeholder(tf.float32)

        # initial learning rate - step width - rate decrement - momentum
        self.__training_param = [0.02, 500, 0.9, 0.9]

    @property
    def model(self):
        try:
            return self.__model
        except AttributeError:
            cnn = CNNFactory.create_cnn(self.__ph_x, [[5, 32], [7, 64]], batch_normalize=True)
            self.__model = MLPFactory.create_mlp(cnn, [512, self.__shape[1]])

        return self.__model

    @property
    def trainer_step(self):
        try:
            return self.__trainer_step
        except AttributeError:
            cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.__ph_y, logits=self.model))

            step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(self.__training_param[0], step, self.__training_param[1], self.__training_param[2], staircase=True)
        
            self.__trainer_step = tf.train.MomentumOptimizer(learning_rate, self.__training_param[3]).minimize(cost_function, global_step=step)
            
            tf.summary.scalar("cost", cost_function)
            tf.summary.scalar("learning_rate", learning_rate)

        return self.__trainer_step
    
    @property
    def evaluator(self):
        try:
            return self.__evaluator
        except AttributeError:
            correct = tf.equal(tf.argmax(self.model, axis=(1)), tf.argmax(self.__ph_y, axis=(1)))
            self.__evaluator = tf.reduce_sum(tf.to_float(correct))

        return self.__evaluator

    def get_feed_dict(self, x_array, y_array, train=False):
        if train:
           return {self.__ph_x: x_array, self.__ph_y: y_array, self.__keepprob: 0.99}
        return {self.__ph_x: x_array, self.__ph_y: y_array, self.__keepprob: 1.0}
