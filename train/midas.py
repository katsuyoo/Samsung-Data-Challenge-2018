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
        self.train_batch = train_batch
        self.dropout_level = dropout_level
        

def modeling(self,
                origin_data,
                num_levels
                ):
        
        self.na_matrix = origin_data.notnull().astype(np.bool)
        self.origin_data = origin_data.fillna(0)
        
        size_input_attribute = self.origin_data.shape[1]
        
        
        # add additional data size to self.encoder_layers attribute.
        # vice versa, for self.decoder_layers does.
        ####################################################################################################check point for add
        self.encoder_layers.insert(0, size_input_attribute)
        self.decoder_layers.append(size_input_attribute)
    
        # Build graph
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            #Placeholders
            self.X = tf.placeholder(tf.float32, [None, size_input_attribute])
            self.na = tf.placeholder(tf.bool, [None, size_input_attribute])

            #Instantiate and initialise variables
            w = []
            b = []
            ow = []
            ob = []
            
            #encoder variables
            for n in range(len(self.encoder_layers) -1):
                w.append(tf.Variable(tf.truncated_normal([self.encoder_layers[n], self.encoder_layers[n+1]], mean = 0, stddev = 1.0 / np.sqrt(in_size + add_size))))
                b.append(tf.Variable(tf.zeros([self.encoder_layers[n+1]])))
            
            #decoder variables
            for n in range(len(self.decoder_layers) -1):
                ow.append(tf.Variable(tf.truncated_normal([self.decoder_layers[n], self.decoder_layers[n+1]], mean = 0, stddev = 1.0 / np.sqrt(self.decoder_layers[0]))))
                ob.append(tf.Variable(tf.zeros([self.decoder_layers[n+1]])))
 
            #Build the neural network
            def encoder(X):
                for i in range(len(self.encoder_layers) -1):
                    if (i == 0):
                        X_dropout = tf.nn.dropout(X, self.input_dropout)
                        X = tf.matmul(X_dropout, w[i]) + b[i]
                        X = tf.nn.elu(X)
                    else:
                        X_dropout = self.build_layer(X, w[i], b[i], dropout_rate = self.hidden_dropout)
                        X = tf.matmul(X_dropout, w[i]) + b[i]
                        X = tf.nn.elu(X)
                return X
            
            def decoder(X):
                for i in range(len(self.decoder_layers) -1):
                    if (i == len(self.decoder_layers) - 2):
                        X_dropout = tf.nn.dropout(X, self.hidden_dropout)
                        X = tf.matmul(X_dropout, ow[i]) + ob[i]
                    else:
                        X_dropout = tf.nn.dropout(X, self.hidden_dropout)
                        X = tf.matmul(X_dropout, ow[i]) + ob[i]
                        X = tf.nn.elu(X)
                
                X = tf.split(X, num_levels, axis = 1)
                return X
                
                ################################### likelihood missing issue #####################################        

            def output(x):
                output = []
                j = 0
                for i in num_levels:
                    if i == 1:
                        output.append(x[j])
                        j += 1
                    else:
                        output.append(tf.nn.softmax(x[j]))
                        j += 1
                return tf.concat(output, axis= 1)                     

            # variables for data generating (forward propagation)
            manifold = encoder(self.X)

            generated_x = decoder(manifold)
            self.output = output(generated_x)

            # variables for loss (backward propagation)
            x = self.X
            concated_num_levels = num_levels

            x_split = tf.split(x, concated_num_levels, axis = 1)    
            na_split = tf.split(self.na, concated_num_levels, axis = 1)         
            
            #L2 loss / KL-Divergence                             
            cost_list_mse = []
            cost_list_ce = []
                                    
            l2_penalty=tf.multiply(tf.reduce_mean([tf.nn.l2_loss(w) for w in w]+[tf.nn.l2_loss(w) for w in ow]), 0.05)
                   
            j = 0
            for i in concated_num_levels:
                if i == 1:
                    # case for numeric
                    cost_list_mse.append(tf.sqrt(tf.losses.mean_squared_error(tf.boolean_mask(x_split[j], na_split[j]), tf.boolean_mask(generated_x[j], na_split[j]))))
                    j += 1
                        
                else:
                    # case for categorical
                    cost_list_ce.append(tf.losses.softmax_cross_entropy(tf.reshape(tf.boolean_mask(x_split[j], na_split[j]), [-1, i]), tf.reshape( tf.boolean_mask(generated_x[j], na_split[j]), [-1, i]) )) 
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

        self.origin_data.reset_index(drop=True,inplace=True)
        na_loc = self.na_matrix.values

        with tf.Session(graph= self.graph) as sess:
            sess.run(self.init)
            for epoch in range(epochs):
                count = 0
                run_loss = 0
                run_ce = 0
                run_mse = 0
                run_l2p = 0
                
                batch_pick = np.random.permutation(np.arange(self.origin_data.shape[0]))
                for batch in range(0,self.origin_data.shape[0],self.train_batch):
                    feed_list = []
                    #if np.sum(batch[1]) == 0:
                    #    continue
                    feed_list.append(self.origin_data.loc[batch_pick[batch:batch+16]])
                    feed_list.append(na_loc[batch_pick[batch:batch+16]])
                    feedin = {self.X: feed_list[0], self.na: feed_list[1]}
                    
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

               