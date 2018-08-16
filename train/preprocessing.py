from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


class preprocessing(object):
    
    def __init__(
                 self,
                 data_path
                 ):
        
        
        def missing_generation(data):
        def onehot_convert(data):
        def minmax_scaling(data):
           
            
            
        self.train_data = pd.read_csv('adult_data.csv')
        missing_generation(self.train_data)
        self.cat_columns = onehot_convert(self.train_data)
        minmax_scaling(self.train_data)
