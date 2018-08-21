import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse

class Midas(object):

    def __init__(self,
               encoder_layers= [256, 256, 256],
               decoder_layers= 'reversed',
               learn_rate= 1e-4,
               input_drop= 0.8,
               train_batch = 16,
               savepath= 'tmp/MIDAS',
               seed= None,
               loss_scale= 1,
               init_scale= 1,
               latent_space_size = 4,
               cont_adj= 1.0,
               softmax_adj= 1.0,
               dropout_level = 0.5,
               weight_decay = 'default',
               vae = True,
               vae_alpha = 1.0
               ):
        
        self.encoder_layers = encoder_layers
        
        if decoder_layers == 'reversed':
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
        self.seed = None
        self.loss_scale = loss_scale
        self.init_scale = init_scale
        self.latent_space_size = latent_space_size
        self.dropout_level = dropout_level
        self.prior_strength = vae_alpha
        self.cont_adj = cont_adj
        self.softmax_adj = softmax_adj
        self.vae = vae

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

    def _batch_iter_output(self,
                  train_data,
                  b_size = 256):
        # function for batch, used in test phase

        indices = np.arange(train_data.shape[0])
        for start_idx in range(0, train_data.shape[0], b_size):
            excerpt = indices[start_idx:start_idx + b_size]
            if self.additional_data is None:
                yield train_data[excerpt]
            else:
                yield train_data[excerpt], self.additional_data.values[excerpt]

    def _build_layer(self,
                   X,
                   weight_matrix,
                   bias_vec,
                   dropout_rate = 0.5,
                   output_layer= False):
        
        X_tx = tf.matmul(tf.nn.dropout(X, dropout_rate), weight_matrix) + bias_vec
        
        if output_layer:
            return X_tx
        else:
            return tf.nn.elu(X_tx)

    def _build_variables(self,
                       weights,
                       biases,
                       num_in,
                       num_out,
                       scale= 1):
    
        weights.append(tf.Variable(tf.truncated_normal([num_in, num_out],
                                                       mean = 0,
                                                       stddev = scale / np.sqrt(num_in + num_out))))
        biases.append(tf.Variable(tf.zeros([num_out]))) #Bias can be zero:/


    def build_model(self,
                imputation_target,
                num_levels,
                additional_data = None,
                verbose= True,
                ):
        print("num_levels :")
        print(num_levels)
        self.na_matrix = imputation_target.notnull().astype(np.bool)
        self.imputation_target = imputation_target.fillna(0)
        if additional_data is not None:
            self.additional_data = additional_data.fillna(0)
        
        in_size = self.imputation_target.shape[1]

        if additional_data != None:
              add_size = self.additional_data.shape[1]
        else:
              add_size = 0
        
        # add additional data size to self.encoder_layers attribute.
        # vice versa, for self.decoder_layers does.
        self.encoder_layers.insert(0, in_size + add_size)
        self.decoder_layers.append(in_size)
        #print(self.encoder_layers)
        #print(self.decoder_layers)

        #Build graph
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            #Placeholders
            self.X = tf.placeholder(tf.float32, [None, in_size])
            self.na_idx = tf.placeholder(tf.bool, [None, in_size])
            #self.latent_inputs = tf.placeholder(tf.float32, [None, self.latent_space_size])

            if additional_data != None:
                self.X_add = tf.placeholder(tf.float32, [None, add_size])

            #Instantiate and initialise variables
            _w = []
            _b = []
            _zw = []
            _zb = []
            _ow = []
            _ob = []
            dict_for_vae_weights = {}
            for i in range(len(num_levels)):
                # one DNN for one variables.
                dict_for_vae_weights["_ow{0}".format(i)] = []
                dict_for_vae_weights["_ob{0}".format(i)] = []


            #encoder variables
            for n in range(len(self.encoder_layers) -1):
                self._build_variables(weights= _w, biases= _b,
                                               num_in= self.encoder_layers[n],
                                               num_out= self.encoder_layers[n+1],
                                               scale= self.init_scale)

            #Latent variables mean, variance(for vae modeling)
            if (self.vae == True):
                self._build_variables(weights= _zw, biases= _zb,
                                                 num_in= self.encoder_layers[-1],
                                                 num_out= self.latent_space_size*2,
                                                 scale= self.init_scale)
                self._build_variables(weights= _zw, biases= _zb,
                                                 num_in= self.latent_space_size,
                                                 num_out= self.decoder_layers[0],
                                                 scale= self.init_scale)

            #decoder variables
            for n in range(len(self.decoder_layers) -1):
                self._build_variables(weights= _ow, biases= _ob,
                                                   num_in= self.decoder_layers[n],
                                                   num_out= self.decoder_layers[n+1])


            #decoder variables for vae networks, one DNNs for one attribute.
            # for numerical variables, output neurons : mu, log_sigma
            # for categorical variables, output neurons : R-dimensional vectors, representing vector of unnormalized probabilities.
            for i in range(len(num_levels)):
                for j in range(len(self.decoder_layers) - 1):
                    if (j == len(self.decoder_layers) - 2):
                        # if reached (output layers - 1)th layer, make connections to latent space, and sample x from latent space.
                        # therefore, 1. weights and biases for (output layers -1)th layer and latent spaces
                        # 2. weights and biases for latent spaces to x are needed.
                        if num_levels[i] == 1:
                            # if integer variables, outputs neurons are mu, log_sigma.
                            self._build_variables(weights= dict_for_vae_weights["_ow{0}".format(i)], biases= dict_for_vae_weights["_ob{0}".format(i)],
                                          num_in = self.decoder_layers[j], num_out=2)
                            # erase following weights and biases, because we choosed sampling procedure, as reparametrization trick.
                            # self._build_variables(weights= dict_for_vae_weights["_ow{0}".format(i)], biases= dict_for_vae_weights["_ob{0}".format(i)],
                            #              num_in = 2, num_out=1)

                        else:
                            # if categorical variables, outputs neurons are R-dimensional vectors, representing vector of unnormalized probabilities.
                            self._build_variables(weights= dict_for_vae_weights["_ow{0}".format(i)], biases= dict_for_vae_weights["_ob{0}".format(i)],
                                          num_in = self.decoder_layers[j], num_out=num_levels[i])
                            # self._build_variables(weights= dict_for_vae_weights["_ow{0}".format(i)], biases= dict_for_vae_weights["_ob{0}".format(i)],
                            #             num_in = num_levels[i], num_out=num_levels[i])
                    else:
                        self._build_variables(weights= dict_for_vae_weights["_ow{0}".format(i)], biases= dict_for_vae_weights["_ob{0}".format(i)],
                                          num_in = self.decoder_layers[j], num_out=self.decoder_layers[j+1])

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

            def encoder_to_z(X):
                X = self._build_layer(X, _zw[0], _zb[0], dropout_rate = self.dropout_level, output_layer= True)
                x_mu, x_log_sigma = tf.split(X, [self.latent_space_size]*2, axis= 1)
                return x_mu, x_log_sigma

            def sample_z(x_mu, x_log_sigma):
                epsilon = tf.random_normal(tf.shape(x_mu))
                z = x_mu + epsilon * tf.exp(x_log_sigma)
                kld = 1/2 * (tf.reduce_sum(1 + 2*x_log_sigma - x_mu**2 - tf.exp(2*x_log_sigma), axis=1)*self.prior_strength * - 0.5 )
                return z, kld

            def z_to_decoder(z):
                X = self._build_layer(z, _zw[1], _zb[1], dropout_rate = self.dropout_level)
                return X

            def decoder(X, origin_X = None):
                # decoder functions returns lists of generated X.
                # therefore, if the self.vae = True, sampling phases are needed and, this function include sampling process.
                # it self.vae = True, this function must return the likelihood values. 
                if self.vae == True:
                    generated_X = []
                    likelihood = []
                    print(X)
                    print(origin_X)
                    for i in range(len(num_levels)):
                        temp_X = []
                        for j in range(len(self.decoder_layers) -1):
                            if (j == len(self.decoder_layers) - 2):
                                if num_levels[i] == 1:
                                    # numeric case, gaussian likelihood
                                    temp_X.append(self._build_layer(temp_X[-1], dict_for_vae_weights["_ow{0}".format(i)][j], dict_for_vae_weights["_ob{0}".format(i)][j], dropout_rate = self.dropout_level, output_layer = True))
                                    x_mu, x_log_sigma = tf.split(temp_X[-1], [1, 1], axis = 1)
                                    # tf.subtract supports broadcast
                                    likelihood.append(tf.reduce_sum(1 / tf.sqrt(2 * np.pi * tf.exp(x_log_sigma)) * tf.exp( - tf.subtract(origin_X[i], x_mu))**2 / (2 * (tf.exp(x_log_sigma)**2) ) ) )
                                    # loss update is done through likelihood, output is made of just connection of neurons.
                                    epsilon = tf.random_normal(tf.shape(x_mu))
                                    sampled_x = x_mu + epsilon * tf.exp(x_log_sigma)
                                    generated_X.append(sampled_x)
                                    break        
                                else:
                                     # categorical case, softmax ( -x[original_X's idx] )
                                     # categorical case doesn't need sampling(maybe. ) just take softmax. << 약간 어색한데..
                                     # in paper, there is no notification about sampling of categorical data...
                                    temp_X.append(self._build_layer(temp_X[-1], dict_for_vae_weights["_ow{0}".format(i)][j], dict_for_vae_weights["_ob{0}".format(i)][j], dropout_rate = self.dropout_level, output_layer = True))
                                    print("origin_X")
                                    print(origin_X)
                                    idx_origin_X = tf.argmax(origin_X[i], axis = 1)
                                    print("idx")
                                    print(idx_origin_X)
                                    numerator_X = []
                                     
                                    
                                    for k in range(num_levels[i]):
                                        print("temp_X")
                                        print(temp_X[-1])
                                        print(temp_X[-1][k])
                                        numerator_X.append(tf.slice(temp_X[-1][k], [k, idx_origin_X[k]], [1, 1]))
                        
                                        
                                    likelihood.append(tf.reduce_sum(tf.exp(tf.negative(numerator_X)) / tf.reduce_sum(tf.exp(tf.negative(temp_X[-1])), axis = 1)))
                                    generated_X.append(temp_X[-1])
                                    break

                            else:
                                if temp_X == []:
                                    temp_X.append(self._build_layer(X, dict_for_vae_weights["_ow{0}".format(i)][j], dict_for_vae_weights["_ob{0}".format(i)][j], dropout_rate = self.dropout_level, output_layer = True))
                                else:
                                    temp_X.append(self._build_layer(temp_X[-1], dict_for_vae_weights["_ow{0}".format(i)][j], dict_for_vae_weights["_ob{0}".format(i)][j], dropout_rate = self.dropout_level, output_layer = True))

                    print(likelihood)
                   
                    return generated_X, tf.reduce_prod(likelihood)


                else:
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


            # if self.vae = True, construct vae model
            if (self.vae == True):                                 
                if self.additional_data != None:
                    x_temp = tf.concat([self.X, self.X_add], axis = 1)

                else:
                    x_temp = self.X
                encoded_x = encoder(x_temp)
                x_mu, x_log_sigma = encoder_to_z(encoded_x)
                generated_z, kld = sample_z(x_mu, x_log_sigma)
                generated_x, likelihood = decoder(z_to_decoder(generated_z), x_temp)
                output = output_function(generated_x)

            # if self.vae = False, construct mida model                         
            else:
                if self.additional_data != None:
                    encoded_x = encoder(tf.concat([self.X, self.X_add], axis= 1))
                else:
                    encoded_x = encoder(self.X)

                generated_x = decoder(encoded_x)
                output = output_function(generated_x)

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
                lmbda = 1/self.imputation_target.shape[0]
            else:
                lmbda = self.weight_decay

            # loss for vae model                         
            if self.vae:
                l2_penalty = tf.multiply(tf.reduce_mean(
                    [tf.nn.l2_loss(w) for w in _w]+\
                    [tf.nn.l2_loss(w) for w in _zw]+\
                    [tf.nn.l2_loss(w) for w in _ow]
                    ), lmbda)                          
              
                print("l2_penalty")
                print(l2_penalty)
                print("kl divergence")
                print(kld)
                print("likelihood")
                print(likelihood)
                
                # self.joint_loss = likelihood + l2_penalty + kld
                self.likelihood = likelihood
                self.joint_loss = likelihood
                #self.joint_loss = likelihood
                #self.joint_loss = tf.reduce_mean(likelihood + kld + l2_penalty)
            # loss for mida model                        
            else:
                l2_penalty = tf.multiply(tf.reduce_mean(
                    [tf.nn.l2_loss(w) for w in _w]+\
                    [tf.nn.l2_loss(w) for w in _ow]
                    ), lmbda)
                    
                j = 0
                for i in concated_num_levels:
                    na_adj = tf.cast(tf.count_nonzero(na_split[j]),tf.float32) / tf.cast(tf.size(na_split[j]),tf.float32)
                    print(na_adj)
                    if i == 1:
                        cost_list_mse.append(tf.sqrt(tf.losses.mean_squared_error(tf.boolean_mask(x_split[j], na_split[j]), tf.boolean_mask(generated_x[j], na_split[j]))) * self.cont_adj * na_adj)
                        j += 1
                        
                    else:
                        print(x_split[j])
                        print( tf.boolean_mask(x_split[j], na_split[j] ) )
                                                             
                        cost_list_ce.append(tf.losses.softmax_cross_entropy( tf.reshape(tf.boolean_mask(x_split[j], na_split[j]), [-1, i]), tf.reshape( tf.boolean_mask(generated_x[j], na_split[j] ), [-1, i]) )  *self.softmax_adj *na_adj ) 
                        j += 1

                self.ce = tf.reduce_sum(cost_list_ce)
                self.mse = tf.reduce_sum(cost_list_mse)
                self.l2p = l2_penalty
                self.joint_loss = self.ce + self.mse + self.l2p
                #version 2, original midas
                #self.joint_loss = tf.reduce_mean(tf.reduce_sum(cost_list) + l2_penalty + kld)


            self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.joint_loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        return self

    def train_model(self,
                  training_epochs= 100,
                  verbose= True,
                  verbosity_ival= 1,
                  excessive= False):


        if self.seed is not None:
            np.seed(self.seed)

        feed_data = self.imputation_target.values
        na_loc = self.na_matrix.values

        with tf.Session(graph= self.graph) as sess:
            sess.run(self.init)
            if verbose:
                print("Model initialised")
            for epoch in range(training_epochs):
                count = 0
                run_loss = 0
                run_ce = 0
                run_mse = 0
                run_l2p = 0
                for batch in self._batch_iter(feed_data, na_loc, self.train_batch):
                    if np.sum(batch[1]) == 0:
                        continue
                    feedin = {self.X: batch[0], self.na_idx: batch[1]}
                    
                    #likelihood = sess.run([self.likelihood], feed_dict = feedin)
                    #print(likelihood)
                    
                    if self.additional_data is not None:
                        feedin[self.X_add] = batch[2]
                    loss,  _ = sess.run( [self.joint_loss, self.train_step] , feed_dict= feedin)
                    mse = sess.run(self.mse, feed_dict = feedin)
                    ce = sess.run(self.ce, feed_dict = feedin)
                    l2p = sess.run(self.l2p, feed_dict = feedin)

                    if excessive:
                        print("Current cost:", loss)
                    count +=1
                    if not np.isnan(loss):
                        run_loss += loss
                        run_ce += ce
                        run_mse += mse
                        run_l2p += l2p
                if verbose:
                    if epoch % verbosity_ival == 0:
                        #print('Epoch:', epoch, ", loss:", str(run_loss/count))
                        print('Epoch:', epoch, ", loss:", str(run_loss/count), ", mse:", str(run_mse/count), ", ce:", str(run_ce/count), ", l2p:", str(run_l2p/count))
                print("Training complete. Saving file...")
                save_path = self.saver.save(sess, self.savepath)
                print("Model saved in file: %s" % save_path)
            return self

    def batch_generate_samples(self,
                             test_data,
                             m= 50,
                             b_size= 256,
                             verbose= True):

        # This function is for testing model.
        # this function returns output_list of given data through this model !!
        # **data_afteration needed. maybe cross-validation with this result?

        idx_null = test_data.notnull()
        test_data = test.data.fillna(0)

        self.output_list = []

        with tf.Session(graph= self.graph) as sess:
            self.saver.restore(sess, self.savepath)
            if verbose:
                print("Model restored.")
            for n in range(m):
                feed_data = test_data.values
                minibatch_list = []
                for batch in self._batch_iter_output(test_data, b_size):
                    feedin = {self.X: batch}
                    y_batch = pd.DataFrame(sess.run(output, feed_dict= feedin), columns= test_data.columns)
                    minibatch_list.append(y_batch)
                y_out = pd.DataFrame(pd.concat(minibatch_list, ignore_index= True), columns= test_data.columns)

                # ignore non-missing column of test data sample, only insert values on missing column on test_data.
                output_df = test_data.copy()
                output_df[np.invert(idx_null)] = y_out[np.invert(idx_null)]
                self.output_list.append(output_df)
        return self

               
  ########################### consideration : overimpute #############################
