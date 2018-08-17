from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


class dataProcessor:
    
    def __init__(self):
        return
    
    def missing_generation(self):
            print('generate missing parts')
            all_atrr = ['주야','요일','사망자수',
                        '사상자수','중상자수','경상자수','부상신고자수','발생지시도',
                        '발생지시군구','사고유형_대분류','사고유형_중분류','법규위반',
                        '도로형태_대분류','도로형태','당사자종별_1당_대분류',
                        '당사자종별_2당_대분류']
            self.train_data = self.train_data[all_atrr]
            self.train_ref = self.train_ref[all_atrr]
            np.random.seed(120)
            missing_index = pd.DataFrame(np.zeros_like(self.train_data), 
                                         columns=self.train_data.columns)
            
            for row in range(missing_index.shape[0]):
                #subset = np.random.randint(missing_index.shape[1],
                #                          size=np.random.randint(low=2,high=5,size=1))
                rand_pick = np.random.permutation(self.train_data.columns)
                for i in rand_pick[0:3]:
                    #print([row, i])
                    missing_index.loc[row, i]=1
                #missing_index.loc[row, subset] = 1
            self.train_data[missing_index==1] = np.nan
            #print(missing_index)
            #print(self.train_data)
    
        
    def onehot_convert(self,
                      isTrain):
        categorical = ['주야','요일','발생지시도','발생지시군구','사고유형_대분류',
                          '사고유형_중분류','법규위반','도로형태_대분류','도로형태',
                          '당사자종별_1당_대분류','당사자종별_2당_대분류']
        print('convert to onehot data')
        if isTrain:
            self.column_dtypes=[]
            #self.train_data.columns.str.strp()
            cat_data = self.train_data[categorical]
            self.train_data.drop(categorical, axis=1, inplace=True)
            constructor_list = [self.train_data]
            for column in range(self.train_data.shape[1]):
                self.column_dtypes.append(1)

            for column in cat_data.columns:
                index = cat_data[column].isnull()
                temp = pd.get_dummies(cat_data[column])
                temp[index] = np.nan
                constructor_list.append(temp)
                self.column_dtypes.append(temp.shape[1])

            self.train_data = pd.concat(constructor_list, axis=1)
        else:
            cat_data = pd.concat([self.test_data[categorical],self.train_ref[categorical]],
                                ignore_index=True)
            self.test_data.drop(categorical, axis=1, inplace=True)
            constructor_list = [self.test_data]

            for column in cat_data.columns:
                index = cat_data[column].isnull()
                temp = pd.get_dummies(cat_data[column])
                temp[index] = np.nan
                temp.drop(range(self.test_data[categorical].shape[0],cat_data.shape[0]))
                constructor_list.append(temp)

            self.test_data = pd.concat(constructor_list, axis=1)



    def minmax_scaling(self,
                      isTrain=True):
        print('0 to 1 scaling')
        scaler = MinMaxScaler()
        
        if isTrain:
            index = self.train_data.isnull()
            self.train_data.fillna(0, inplace = True)
            self.train_data = pd.DataFrame(scaler.fit_transform(self.train_data),
                                            columns= self.train_data.columns)
            self.train_data[index] = np.nan
        else:
            index = self.test_data.isnull()
            self.test_data.fillna(self.test_data.median(), inplace = True)
            self.test_data = pd.DataFrame(scaler.fit_transform(self.test_data),
                                            columns= self.test_data.columns)
            self.train_data[index] = np.nan


    def preProcessing(self,
                     data_path,
                     isTrain):
        
        if isTrain:
            self.train_data = pd.read_csv(data_path).loc[:5]
            self.train_ref = self.train_data.copy()
            self.missing_generation()
        else:
            self.test_data = pd.read_csv(data_path)

        self.onehot_convert(isTrain)
        self.minmax_scaling()
        
        print(self.train_data.head())
        if isTrain:
            return self.train_data, self.column_dtypes
        else:
            return self.test_data
            
        #def postProcessing(self,
        #                  row_pred):
        #    
        #    
        #    return self.pred_data
            
            
            
        
        
