from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


class dataProcessor:
    
    def __init__(self):
        return
    
    def missing_generation(self):
            print('generate missing parts')
            self.all_atrr = ['주야','요일','사망자수',
                        '사상자수','중상자수','경상자수','부상신고자수','발생지시도',
                        '발생지시군구','사고유형_대분류','사고유형_중분류','법규위반',
                        '도로형태_대분류','도로형태','당사자종별_1당_대분류',
                        '당사자종별_2당_대분류']
            self.train_data = self.train_data[self.all_atrr]
            self.train_ref = self.train_ref[self.all_atrr]
            np.random.seed(120)
            self.missing_index = pd.DataFrame(np.zeros_like(self.train_data), 
                                         columns=self.train_data.columns)
            
            for row in range(self.missing_index.shape[0]):
                rand_pick = np.random.permutation(self.train_data.columns)
                for i in rand_pick[0:3]:
                    self.missing_index.loc[row, i]=1
            self.train_data[self.missing_index==1] = np.nan
    
        
    def onehot_convert(self,
                      isTrain):
        self.categorical = ['주야','요일','발생지시도','발생지시군구','사고유형_대분류',
                          '사고유형_중분류','법규위반','도로형태_대분류','도로형태',
                          '당사자종별_1당_대분류','당사자종별_2당_대분류']
        print('convert to onehot data')
        if isTrain:
            self.column_dtypes=[]
            #self.train_data.columns.str.strp()
            cat_data = self.train_data[self.categorical]
            self.train_data.drop(self.categorical, axis=1, inplace=True)
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
            cat_data = pd.concat([self.test_data[self.categorical],self.train_ref[self.categorical]],
                                ignore_index=True)
            self.test_data.drop(self.categorical, axis=1, inplace=True)
            constructor_list = [self.test_data]

            for column in cat_data.columns:
                index = cat_data[column].isnull()
                temp = pd.get_dummies(cat_data[column])
                temp[index] = np.nan
                temp.drop(range(self.test_data[self.categorical].shape[0],cat_data.shape[0]))
                constructor_list.append(temp)

            self.test_data = pd.concat(constructor_list, axis=1)



    def minmax_scaling(self,
                      isTrain=True):
        print('0 to 1 scaling')
        self.scaler = MinMaxScaler()
        
        if isTrain:
            index = self.train_data.isnull()
            self.train_data.fillna(0, inplace = True)
            self.train_data = pd.DataFrame(self.scaler.fit_transform(self.train_data),
                                            columns= self.train_data.columns)
            self.train_data[index] = np.nan
        else:
            index = self.test_data.isnull()
            self.test_data.fillna(self.test_data.median(), inplace = True)
            self.test_data = pd.DataFrame(self.scaler.fit_transform(self.test_data),
                                            columns= self.test_data.columns)
            self.train_data[index] = np.nan


    def preProcessing(self,
                     data_path,
                     isTrain):
        
        if isTrain:
            self.train_data = pd.read_csv(data_path).loc[:31]
            self.train_ref = self.train_data.copy()
            self.missing_generation()
        else:
            self.test_data = pd.read_csv(data_path)

        self.onehot_convert(isTrain)
        self.minmax_scaling()
        
        if isTrain:
            return self.train_data, self.column_dtypes
        else:
            return self.test_data
            
    def postProcessing(self,
                       data):
        self.numerical=['사망자수','사상자수','중상자수','경상자수','부상신고자수']
        data_num = data[self.numerical]
        data_cat = data.drop(self.numerical,axis=1)
        cat_dtypes = self.column_dtypes[5:]

        data_lists=[]
        sum=0
        for i in range(2):
            temp = data_cat[data_cat.columns[sum:sum+cat_dtypes[i]]]
            temp2= pd.DataFrame(temp.idxmax(axis=1),columns=[self.categorical[i]])
            data_lists.append(temp2)
            sum+=cat_dtypes[i]

        data_lists.append(data_num)

        for j in range(2,len(self.categorical)):
            temp = data_cat[data_cat.columns[sum:sum+cat_dtypes[j]]]
            temp2= pd.DataFrame(temp.idxmax(axis=1),columns=[self.categorical[j]])
            data_lists.append(temp2)
            sum+=cat_dtypes[j]

        self.pred_data = pd.DataFrame(pd.concat(data_lists, axis=1))

                
        return self.pred_data


            
        
        
