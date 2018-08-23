from dae import DAE
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from dataprocessor import dataProcessor


#############################################
#                 Training
#############################################
proc = dataProcessor()
train_data, col_list = proc.preProcessing(data_path='train.csv', isTrain = True)
dae = DAE()
dae.modeling(origin_data = train_data, num_levels = col_list)
dae.training(epochs=5)



#############################################
#                  Testing
#############################################
test_origin = proc.preProcessing(data_path='test_kor.csv', isTrain = False)
dae.restoration(test_data)


test_pred = dae.Y
test_pred_row = proc.postProcessing(test_pred)

result = pd.read_csv('result_kor.csv')
answers = []
for i in range(result.shape[0]):
    answers.append([test_pred_row.loc[result.loc[i,'행']-2,test_pred_row.columns[ord(result.loc[i,'열'])-65]]])

answers = np.asarray(answers)    
answer = pd.DataFrame(answers,colmuns=['값'])
result.drop(,inplace=True)
lists=[]
lists.append(result)
lists.append

answer.to_csv('result_kor.csv')


