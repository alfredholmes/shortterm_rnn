import tensorflow as tf
import csv


class TimeSeriesForcaster:
    #name is a string for the save of the network
    #input length is the number of days in the input for the RNN
    #input size is the dimension of the daily data
    def __init__(self, name, input_length, input_dimension, hidden_layers, learning_rate):
        self.name = name
        self.input_length = input_length
        self.input_dimension = input_dimension
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        #define tensorflow variables
        self.input_tensor = tf.placeholder(tf.float32, [None, input_length, input_dimension])
        self.expected_output_matrix = tf.placeholder(tf.float32, [None, input_dimension])

        #vectors for the RNN
        self.weights = tf.Variable(tf.random_normal([hidden_layers, 2]))
        self.bias = tf.Variable(tf.random_normal([2]))

        #initialize tensorflow
        self.prediction = self.RNN()
        self.error = self.prediction - self.expected_output_matrix
        self.e = tf.reduce_mean(self.error)
        self.cost = tf.reduce_mean(tf.square(self.error))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.saver = tf.train.Saver()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())



    def RNN(self):
        x = tf.unstack(self.input_tensor, axis=1)
        cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_layers)
        output, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
        return tf.sigmoid(tf.matmul(output[-1], self.weights) + self.bias)

    #perform 1 traning iteration
    def basic_train(self, input_data, expected_output, testing_input, testing_output):
        self.session.run(self.optimizer, feed_dict={self.input_tensor: input_data, self.expected_output_matrix: expected_output})
        return self.session.run([self.cost, self.e], feed_dict={self.input_tensor: testing_input, self.expected_output_matrix: testing_output})

    def train_to_minimum(self, error, input_data, expected_output, testing_input, testing_output, tolerance=300):
        min = None
        #tolerance = 300
        steps_without_improvement = 0
        while min is None or min > (1 - error) * self.session.run(self.cost, feed_dict={self.input_tensor: testing_input, self.expected_output_matrix: testing_output}):
            self.session.run(self.optimizer, feed_dict={self.input_tensor: input_data, self.expected_output_matrix: expected_output})
            a = self.session.run(self.cost, feed_dict={self.input_tensor: testing_input, self.expected_output_matrix: testing_output})
            if min is None or min > a:
                min = a
                print(a)
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                print('\t' + str(a) + ' ' +str(steps_without_improvement))
            if steps_without_improvement > tolerance:
                return


    def get_error_for_metric(self, id, test_input, expected_output):
        error = self.session.run(self.error, feed_dict={self.input_tensor: test_input, self.expected_output_matrix: expected_output})
        column = self.session.run(tf.transpose(error)[id])
        return self.session.run([tf.reduce_mean(column), tf.reduce_mean(tf.square(column))])


    def predict(self, input_data):
        return self.session.run(self.prediction, feed_dict={self.input_tensor: input_data})

    def save(self):
        self.saver.save(self.session, './save/' + self.name)
    def restore(self):
        self.saver.restore(self.session, './save/' + self.name)
