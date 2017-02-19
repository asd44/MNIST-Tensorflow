import tensorflow as tf


class MLPFactory:
    @staticmethod
    def normal_var(shape, name=None, dtype=None):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name, dtype=dtype)

    @staticmethod
    def batch_normalizer(x):
        input_shape = x.get_shape().as_list()[1:]
        offset = MLPFactory.normal_var(input_shape)
        scale = tf.Variable(tf.ones(input_shape))

        mean, variance = tf.nn.moments(x, [0])
        return tf.nn.batch_normalization(x, mean, variance, offset, scale, 1e-3)

    @staticmethod
    def fully_connected_layer(x, shape):
        fcl_w = MLPFactory.normal_var(shape)
        fcl_b = MLPFactory.normal_var([shape[1]])
        return tf.matmul(x, fcl_w) + fcl_b

    @staticmethod
    def fully_connected_normalized_layer(x, shape):
        fcl_w = MLPFactory.normal_var(shape)
        return MLPFactory.batch_normalizer(tf.matmul(x, fcl_w))

    @staticmethod
    def create_mlp(input, layers, activation=tf.nn.relu, normalize=False):
        ''' that's shitty but i don't want to import np just for that '''
        current_synaps = 1
        for i in input.get_shape().as_list()[1:]:
            current_synaps *= i

        if normalize:
            fcl = MLPFactory.fully_connected_normalized_layer
        else:
            fcl = MLPFactory.fully_connected_layer

        current_tensor = tf.reshape(input, [-1, current_synaps])
        for params in layers[:-1]:
            current_tensor = activation(fcl(current_tensor, [current_synaps, params]))
            current_synaps = params

        return MLPFactory.fully_connected_layer(current_tensor, [current_synaps, layers[-1]])


class MLP:
    def __init__(self, shape, x_dtype, y_dtype):
        self.__shape = shape
        self.__ph_x = tf.placeholder(x_dtype, shape=[None, shape[0][0], shape[0][1], shape[0][2]])
        self.__ph_y = tf.placeholder(y_dtype, shape=[None, shape[1]])

        #initial learning rate - step width - rate decrement
        self.__training_param = [1.0, 500, 0.9]
        
    @property
    def model(self):
        try:
            return self.__model
        except AttributeError:
            self.__model = MLPFactory.create_mlp(self.__ph_x, [128, 128, self.__shape[1]], normalize=True)

        return self.__model

    @property
    def trainer_step(self):
        try:
            return self.__trainer_step
        except AttributeError:
            cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.__ph_y, logits=self.model))

            step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(self.__training_param[0], step, self.__training_param[1], self.__training_param[2], staircase=True)

            self.__trainer_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function, global_step=step)

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
        return {self.__ph_x: x_array, self.__ph_y: y_array}
