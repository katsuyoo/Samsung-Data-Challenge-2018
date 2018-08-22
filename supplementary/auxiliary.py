import tensorflow as tf
import numpy as np

class auxiliary_net:

    def __init__(self, session, input_size, output_size, data_type, no_levels = None, name = "auxiliary_net"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.data_type = data_type
        self.no_levels = no_levels
        self.net_name = name

        self.build_network()



    def build_network(self, h_size = 16, l_rate = 1e-3, p = 0.05, reg = 0.05):
        
        # network (computational graph)
        with tf.variable_scope(self.net_name):
            self.X = tf.placeholder(tf.float32, [None, self.input_size], name = "input_x")

            #input dropout
            noisy_input_x = tf.layers.dropout(self.X, rate = p, noise_shape = [1, self.input_size], name = "noisy_input_x")

            #first layer of weights
            W1 = tf.get_variable("W1", shape = [self.input_size, h_size], initializer= tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", shape = [h_size], initializer= tf.contrib.layers.xavier_initializer())
            h1 = tf.nn.relu(tf.matmul(self.noisy_input_x, W1) + b1)

            """
            #second layer of weights
            W2 = tf.get_variable("W2", shape = [h_size, h_size], initializer= tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", shape = [h_size], initializer= tf.contrib.layers.xavier_initializer())
            h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
            """

            #third layer of weights
            W2 = tf.get_variable("W2", shape = [h_size, self.output_size], initializer= tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", shape = [self.output_size], initializer= tf.contrib.layers.xavier_initializer())
            self.Y_hat = tf.matmul(h1, W2) + b2      #network output


        self.Y = tf.placeholder(shape = [None, self.output_size], dtype = tf.float32)

        #Loss function
        if self.data_type == "real-valued":
            # Mean squared error
            self.loss = tf.reduce_mean(tf.square(self.Y - self.Y_hat), axis = 0)

        elif self.data_type == "categorical":
            # cross entropy
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.Y, logits = self.Y_hat), axis = 0)

        elif self.data_type == "ordinal":
            
            # error tailored to ordinal type data (reference: "Loss Functions for Preference Levels: Regression with Discrete Ordered Labels" by Rennie & Srebro, 2005)
            mu_2toK_1 = tf.get_variable("mu", shape = [self.no_levels-2], initializer = tf.contrib.layers.xavier_initializer())  #we have to fix the extreme ends: mu_1 = -1, mu_K = 1 
            mu_1 = tf.constant([-1], tf.float32)
            mu_K = tf.constant([1], tf.float32)
            mu = tf.concat([mu_1, mu_2toK_1, mu_K], axis = 0)

            pi = tf.get_variable("pi", shape = [self.no_levels], initializer = tf.contrib.layers.xavier_initializer())

            # network output: shape = [N, 1]
            Z = self.Y_hat
            
            # log p(Y, Z) where array(i,j) = log p(y = j | Zi): shape = [N, K]
            log_joint_prob = tf.multiply(Z, mu) + pi - tf.square(mu) / 2

            # log p(Z) where array(i) = log p(Zi): shape = [N, 1]
            log_marginal_likelihood = tf.reduce_logsumexp(log_joint_prob, axis = 1)

            indices = tf.concat([tf.cumsum(tf.ones_like(self.Y, dtype = tf.int32), axis = 0) - 1, tf.cast(self.Y, dtype = tf.int32)], axis = 1)

            # log p(y = Yi | Zi): shape = [N, 1]
            log_joint_prob_i = tf.gather_nd(log_joint_prob, indices)


            # loss(Y, Z) where array(i) = loss(Yi, Zi): shape = [N, 1]
            loss_i = log_marginal_likelihood - log_joint_prob_i

            # total loss over entire input batch 
            self.loss = tf.reduce_mean(loss_i, axis = 0)


        # L2 regularization
        self.regularized_loss = self.loss + reg * (tf.reduce_sum(tf.reduce_sum(tf.square(W1), axis = 1), axis = 0) + tf.reduce_sum(tf.reduce_sum(tf.square(W2), axis = 1), axis = 0))

        # Learning
        self.train = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self.regularized_loss)



    def predict(self, x_test, y_test):
        y_pred, test_loss = self.session.run([self.Y_hat, self.loss], feed_dict = {self.X: x_test, self.Y: y_test})          # type np.array of shape = [N, self.output_size]

        if self.data_type == "real-valued":
            return y_pred, test_loss                                            # test_loss is mean squared error over test data
        
        elif self.data_type == "categorical":
            argmax_idx = np.argmax(y_pred, axis = 1)                            # shape = [N,]
            one_hot_pred = np.zeros(y_pred.shape)           
            one_hot_pred[np.arange(y_pred.shape[0]), argmax_idx] = 1            # one-hot encoding
            
            comparison = np.equal(one_hot_pred, y_test).astype(int)
            accuracy = np.mean(np.sum(comparison, axis = 1))
            return one_hot_pred, accuracy                                       # classification accuracy
    
        elif self.data_type == "ordinal":
            mse = np.mean(np.square(y_pred - y_test))
            return np.rint(y_pred), mse                                   



    def update(self, x_batch, y_batch):
        return self.session.run([self.loss, self.train], feed_dict = {self.X: x_batch, self.Y: y_batch})