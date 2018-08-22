from midas import Midas
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from dataprocessor import dataProcessor
from math import exp, pow

# import csv2utf8
# already created utf8 files

proc = dataProcessor()
data_0, col_list = proc.preProcessing(data_path='train_utf.csv', isTrain = True)
origin_data = proc.train_ref

rand_pick = np.random.permutation(np.arange(data_0.shape[0]))
missing_index = proc.missing_index.drop(rand_pick[:16])
data_0_test = data_0.loc[rand_pick[:16]]
test_origin = origin_data.loc[rand_pick[:16]]
data_0.drop(rand_pick[:16],inplace=True)

imputer = Midas(vae = False)
imputer.build_model(imputation_target = data_0, num_levels = col_list)

imputer.modeling(training_epochs=1, verbosity_ival= 1)


imputer.batch_generate_samples(data_0_test, m=1,b_size=16)


test_pred = imputer.y_batch



test_pred_row = proc.postProcessing(test_pred)

test_origin_num = test_origin.drop(proc.categorical, axis=1)
test_origin_cat = test_origin[proc.categorical]

test_pred_row_num = test_pred_row.drop(proc.categorical, axis=1)
test_pred_row_cat = test_pred_row[proc.categorical]

missing_index_num = missing_index.drop(proc.categorical, axis=1)
missing_index_cat = missing_index[proc.categorical]


test_origin_cat.reset_index(drop=True, inplace=True)
test_origin_num.reset_index(drop=True, inplace=True)
num_diff = test_origin_num[missing_index_num==1]-test_pred_row_num[missing_index_num==1]
for column in test_origin_num.columns:
    for index in range(test_origin_num.shape[0]):
        num_diff.loc[index, column] = exp(-pow(num_diff.loc[index, column],2))

numerical_score = num_diff.sum().sum()        

cat_diff = pd.DataFrame(np.zeros_like(test_origin_cat), columns=test_origin_cat.columns)
for column in test_origin_cat.columns:
    for index in range(test_origin_cat.shape[0]):
        if test_origin_cat.loc[index, column]==test_pred_row_cat.loc[index, column]:
            cat_diff.loc[index,column]=1
            
categorical_score = cat_diff.sum().sum()



print('Numeric score: ',numerical_score)
print('Categoric score: ',categorical_score)
