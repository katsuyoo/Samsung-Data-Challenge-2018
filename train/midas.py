import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse

class Midas(object):

    def __init__(self,
               encoder_layers= [256, 256, 256],
               learn_rate= 1e-4,
               input_drop= 0.8,
               train_batch = 16,
               savepath= 'tmp/MIDAS',
               dropout_level = 0.5,
               weight_decay = 'default',
               ):
        
        self.encoder_layers = encoder_layers
        self.decoder_layers = self.encoder_layers.copy()
        self.decoder_layers.reverse()
            
        if weight_decay == 'default':
            self.weight_decay = 'default'
        elif type(weight_decay) == float:
            self.weight_decay = weight_decay
                  
        self.learn_rate = learn_rate
        self.input_drop = input_drop
        self.savepath = savepath
        self.additional_data = None
        self.train_batch = train_batch
        self.dropout_level = dropout_level
        
    def _batch_iter(self,
                  train_data,
                  na_mask,
                  b_size = 16):
        # functions for batch, used in train phase
        
        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(0, train_data.shape[0] - b_size + 1, b_size):
            excerpt = indices[start_idx:start_idx + b_size]
        
            if self.additional_data is None:
                yield train_data[excerpt], na_mask[excerpt]
            else:
                yield train_data[excerpt], na_mask[excerpt], self.additional_data.values[excerpt]

    def _build_layer(self,
                   X,
                   w,
                   b,
                   dropout_rate = 0.5,
                   output_layer= False):
        
        X_dropout = tf.nn.dropout(X, dropout_rate)
        X = tf.matmul(X_dropout, w) + b
        
        if output_layer != True:
            return tf.nn.elu(X)
        else:
            return X

    def _build_variables(self,
                       w,
                       b,
                       num_in,
                       num_out):
    
        temp_w = tf.Variable(tf.truncated_normal([num_in, num_out], mean = 0, stddev = scale / np.sqrt(num_in + num_out)))
        temp_b = tf.Variable(tf.zeros([num_out]))
        w.append(temp_w)
        b.append(temp_b) 

    def modeling(self,
                origin_data,
                num_levels,
                additional_data = None,
                verbose= True,
                ):
        
        self.na_matrix = origin_data.notnull().astype(np.bool)
        self.origin_data = origin_data.fillna(0)
        if additional_data is not None:
            self.additional_data = additional_data.fillna(0)
        
        in_size = self.origin_data.shape[1]

        if additional_data != None:
              add_size = self.additional_data.shape[1]
        else:
              add_size = 0
        
        # add additional data size to self.encoder_layers attribute.
        # vice versa, for self.decoder_layers does.
        self.encoder_layers.insert(0, in_size + add_size)
        self.decoder_layers.append(in_size)
      
        # Build graph
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            #Placeholders
            self.X = tf.placeholder(tf.float32, [None, in_size])
            self.na_idx = tf.placeholder(tf.bool, [None, in_size])
            #self.latent_inputs = tf.placeholder(tf.float32, [None, self.latent_space_size])

            if additional_data != None:
                self.X_add = tf.placeholder(tf.float32, [None, add_size])

            #Instantiate and initialise variables
            _w = []
            _b = []
            _ow = []
            _ob = []
            
            #encoder variables
            for n in range(len(self.encoder_layers) -1):
                self._build_variables(weights= _w, biases= _b,
                                               num_in= self.encoder_layers[n],
                                               num_out= self.encoder_layers[n+1]
                                               )

            
            #decoder variables
            for n in range(len(self.decoder_layers) -1):
                self._build_variables(weights= _ow, biases= _ob,
                                                   num_in= self.decoder_layers[n],
                                                   num_out= self.decoder_layers[n+1])


            #Build the neural network
            def encoder(X):
                for n in range(len(self.encoder_layers) -1):
                    if (n == 0):
                        X = self._build_layer(X, _w[n], _b[n],
                                          dropout_rate = self.input_drop)
                    else:
                        X = self._build_layer(X, _w[n], _b[n],
                                          dropout_rate = self.dropout_level)
                return X

            
            def decoder(X, origin_X = None):
                # decoder functions returns lists of generated X.
                # therefore, if the self.vae = True, sampling phases are needed and, this function include sampling process.
                # it self.vae = True, this function must return the likelihood values. 
                
                for n in range(len(self.decoder_layers) -1):
                    if (n == len(self.decoder_layers) - 2):
                        X = self._build_layer(X, _ow[n], _ob[n],
                                              dropout_rate = self.dropout_level, output_layer = True)
                    else:
                        X = self._build_layer(X, _ow[n], _ob[n],
                                              dropout_rate = self.dropout_level)
                generated_X = tf.split(X, num_levels, axis = 1)

                return generated_X
                
                ################################### likelihood missing issue #####################################        

            def output_function(x):
                output_list = []
                j = 0
                for i in num_levels:
                    if i == 1:
                        output_list.append(x[j])
                        j += 1
                    else:
                        output_list.append(tf.nn.softmax(x[j]))
                        j += 1
                return tf.concat(output_list, axis= 1)                     


            if self.additional_data != None:
                encoded_x = encoder(tf.concat([self.X, self.X_add], axis= 1))
            else:
                encoded_x = encoder(self.X)

            generated_x = decoder(encoded_x)
            self.output = output_function(generated_x)

            if self.additional_data != None:
                x = tf.concat([self.X, self.X_add], axis = 1)
                # remind ; additional_data consists of [data , num_levels of add data ]
                concated_num_levels = tf.concat([num_levels, additional_data[1]], axis = 1)

            else:
                x = self.X
                concated_num_levels = num_levels

            na_split = tf.split(self.na_idx, concated_num_levels, axis = 1)         
            x_split = tf.split(x, concated_num_levels, axis = 1)

            #Build L2 loss and KL-Divergence                             
            cost_list_mse = []
            cost_list_ce = []
            # vae
            cost_list = []

            if self.weight_decay == 'default':
                lmbda = 1/self.origin_data.shape[0]
            else:
                lmbda = self.weight_decay
                        
                                    
            l2_penalty = tf.multiply(tf.reduce_mean(
                    [tf.nn.l2_loss(w) for w in _w]+\
                    [tf.nn.l2_loss(w) for w in _ow]
                    ), lmbda)
                    
            j = 0
            for i in concated_num_levels:
                na_adj = tf.cast(tf.count_nonzero(na_split[j]),tf.float32) / tf.cast(tf.size(na_split[j]),tf.float32)
                #print(na_adj)
                if i == 1:
                    cost_list_mse.append(tf.sqrt(tf.losses.mean_squared_error(
                        tf.boolean_mask(x_split[j], na_split[j]), tf.boolean_mask(generated_x[j], na_split[j]))) * na_adj)
                    j += 1
                        
                else:
                    #print(x_split[j])
                    #print( tf.boolean_mask(x_split[j], na_split[j] ) )
                                                             
                    cost_list_ce.append(tf.losses.softmax_cross_entropy(
                        tf.reshape(tf.boolean_mask(x_split[j], na_split[j]), [-1, i]), 
                        tf.reshape( tf.boolean_mask(generated_x[j], na_split[j]), [-1, i]) ) *na_adj ) 
                    j += 1

            self.ce = tf.reduce_sum(cost_list_ce)
            self.mse = tf.reduce_sum(cost_list_mse)
            self.l2p = l2_penalty
            self.joint_loss = self.ce + self.mse + self.l2p
                

            self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.joint_loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        return self

    def training(self, epochs=100):

        feed_data = self.origin_data.values
        na_loc = self.na_matrix.values

        with tf.Session(graph= self.graph) as sess:
            sess.run(self.init)
            for epoch in range(epochs):
                count = 0
                run_loss = 0
                run_ce = 0
                run_mse = 0
                run_l2p = 0
                
                for batch in self._batch_iter(feed_data, na_loc, self.train_batch):
                    if np.sum(batch[1]) == 0:
                        continue
                    feedin = {self.X: batch[0], self.na_idx: batch[1]}
                    
                    if self.additional_data is not None:
                        feedin[self.X_add] = batch[2]
                    loss,  _ = sess.run( [self.joint_loss, self.train_step] , feed_dict= feedin)
                    mse = sess.run(self.mse, feed_dict = feedin)
                    ce = sess.run(self.ce, feed_dict = feedin)
                    l2p = sess.run(self.l2p, feed_dict = feedin)

                    count +=1
                    if not np.isnan(loss):
                        run_loss += loss
                        run_ce += ce
                        run_mse += mse
                        run_l2p += l2p
               
                        
                print('Epoch:', epoch, ", loss:", str(run_loss/count), ", mse:", str(run_mse/count), ", ce:",
                      str(run_ce/count), ", l2p:", str(run_l2p/count))
 
            save_path = self.saver.save(sess, self.savepath)
            
        return self

    def restoration(self,test_data):
        test_data = test_data.fillna(0)
        with tf.Session(graph= self.graph) as sess:
            self.saver.restore(sess, self.savepath)
                
            feed_data = test_data.values
            feedin = {self.X: test_data}
            self.y_batch = pd.DataFrame(sess.run(self.output, feed_dict= feedin), columns= self.origin_data.columns)
            
        return self

               