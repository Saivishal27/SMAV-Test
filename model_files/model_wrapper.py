# Import all the ml classes
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import sys
from compute_metrics import *

from scipy import stats



# Creating & Training Models
class Model:
    def __init__(self,use_cols,labels,window_size,stride):
        # Class level variables initialization

        self._use_cols      = use_cols
        self._labels = labels
        self._stride = stride
        self.__split_percent = 0.25
        # self._target_label  = {'FUEL_RATE':{'Encoding':{},'operating_characteristic_type_name':'measurement'}}

        self._pca=None
        self._scaler=None
        self._model=None
        
        self.__window_size = str(int(window_size/1000))+'S'
        self._train_report=None
        self._test_report=None
        self._hyper_parm_list = None

        self._right_inspect = 6
        self._left_inspect  = -3
        self._right_failure = 7
        self._left_failure = -4

        self._rolling_window_size = 120
        self._asset = 'DG1'

        self._model_cols = ['POWER_MANAGEMENT_SYSTEM/DG1/DG1LUBOILPRESSURE',
                            'POWER_MANAGEMENT_SYSTEM/DG1/DG1ENGINERPM',
                            'POWER_MANAGEMENT_SYSTEM/DG1/DG1COOLANTTEMPERATURE',
                            'POWER_MANAGEMENT_SYSTEM/MSB/DG1_ACTIVE_POWER']
        # self.__split_percent        = split_percent
        # self._x_train_start_time    = train_start
        # self._x_train_end_time      = train_end
        # self._x_test_start_time     = test_start
        # self._x_test_end_time       = test_end

    
    # Train Test split based on the timestamps provided
    def __test_train_split(self, X, Y):
        x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                            train_size= (1 - self.__split_percent),
                                                            test_size= self.__split_percent,
                                                            random_state= 42)

        return x_train, x_test, y_train, y_test

    # Model Training
    def __model_training(self, x_train, x_test, y_train, y_test):
        self._model         = RandomForestRegressor().fit(x_train, y_train)
        y_train_pred        = self._model.predict(x_train)
        y_test_pred         = self._model.predict(x_test)
        rmse                = np.sqrt(mean_squared_error(y_train, y_train_pred))
        r2score             = r2_score(y_train, y_train_pred)
        self._train_report  = {'rmse': rmse, 'r2_score': r2score}
        r2score             = r2_score(y_test, y_test_pred)
        rmse                = np.sqrt(mean_squared_error(y_test, y_test_pred))
        self._test_report   = {'rmse': rmse, 'r2_score'   : r2score}
        self._hyper_parm_list = self._model.get_params()
        self._feature_importances = self._model.feature_importances_

    def pre_process(self,df):
        df = df[df['POWER_MANAGEMENT_SYSTEM/'+self._asset+'/'+self._asset+'ENGINERPM']>=1400]
        print('Data 11\n',df.head())
        print('Data 11 Shape\n',df.shape)
        df = df.dropna()
        print('Data 12\n',df.head())
        print('Data 12 Shape\n',df.shape)
        if len(df)!=0:
            z = np.abs(stats.zscore(df[self._use_cols]))
            df = df[(z < 3).all(axis=1)]
        return df

    # Public function to invoke training
    def train(self, data):
        
        if set(self._use_cols+['time_stamp','asset_no']).issubset(data.columns):
            data = data.set_index(['asset_no','time_stamp'])
            data = self.pre_process(data)
            X   = data[self._model_cols]
            Y   = data[self._labels]
            x_train, x_test, y_train, y_test = self.__test_train_split(X, Y)
            self.__model_training(x_train, x_test, y_train, y_test)
            self._hyper_parm_list.update(dict(zip(self._model_cols, self._feature_importances)))
        else:
            message = "Input data doesn't have all the columns required"
            return (False, message)
      
        return (True, 'Model Trained')

    def check_condition(self,x):
        # print('Data 451:\n')
        onehot = pd.Series([0,0],index = ['DERIVED_'+self._asset+'_LUBRICATION_SYSTEM_FAILURE_WARNING','DERIVED_'+self._asset+'_LUBRICATION_SYSTEM_FAILURE'])
        # print('Data 452:\n')
        if (x>=self._right_inspect and x<self._right_failure) or (x<=self._left_inspect and x>self._left_failure):
            # print('Data 453:\n')
            onehot.loc['DERIVED_'+self._asset+'_LUBRICATION_SYSTEM_FAILURE_WARNING'] = 1
            # print('Data 454:\n')
        elif (x>=self._right_failure) or (x<=self._left_failure):
            # print('Data 455:\n')
            onehot.loc['DERIVED_'+self._asset+'_LUBRICATION_SYSTEM_FAILURE'] = 1
            # print('Data 456:\n')
        # print('Data 457:\n')
        return onehot

    def post_processing(self,data,target_label):
        print('Data 41:\n')
        data[['DERIVED_POWER_MANAGEMENT_SYSTEM/'+self._asset+'/'+self._asset+'LUBOILTEMPERATURE']] = pd.DataFrame(data['prediction'].to_list(),index=data.index)
        print('Data 42:\n')
        data['residue'] = data['POWER_MANAGEMENT_SYSTEM/'+self._asset+'/'+self._asset+'LUBOILTEMPERATURE']-data['DERIVED_POWER_MANAGEMENT_SYSTEM/'+self._asset+'/'+self._asset+'LUBOILTEMPERATURE']
        print('Data 43:\n')
        data.to_csv('D:\SMAV\Backup\Testing_data_DG1.csv')
        temp = data.set_index('time_stamp').groupby(['asset_no'])['residue'].rolling(window = self._rolling_window_size, min_periods=1).mean().reset_index()
        print('Data 44:\n')
        data = data.drop(columns = ['residue']).merge(temp,on=['asset_no','time_stamp'])
        print('Data 45:\n',data.shape)
        data = data.join(data['residue'].apply(self.check_condition))
        print('Data 46:\n',data.shape)
        return data,['DERIVED_POWER_MANAGEMENT_SYSTEM/'+self._asset+'/'+self._asset+'LUBOILTEMPERATURE','DERIVED_'+self._asset+'_LUBRICATION_SYSTEM_FAILURE_WARNING','DERIVED_'+self._asset+'_LUBRICATION_SYSTEM_FAILURE']
    # Public function to invoke the trained model to predict
    def predict(self, data, target_label):
        if set(self._use_cols+['time_stamp','asset_no']).issubset(data.columns):
            print('Data 1:\n',data.head())
            data = data.set_index(['asset_no','time_stamp'])
            print('Data Shape:\n',data.shape)
            data = self.pre_process(data)
            print('Data 2:\n',data.head())
            data = data.reset_index()
            predict_data = data[self._model_cols]
            print('Data 3:\n',predict_data.head())
            if len(predict_data)!=0:
                data['prediction']  = self._model.predict(predict_data)
                print('Data 4:\n',data.head())
                data,target_tags = self.post_processing(data,target_label)
                print('Data 5:\n',data.head())
                return data[['time_stamp','asset_no']+target_tags]
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def test(self,data,metrics):
        if set(self._use_cols+['time_stamp','asset_no']).issubset(data.columns):
            data = data.set_index(['asset_no','time_stamp'])
            data = self.pre_process(data)
            data = data.reset_index()
            X   = data[self._model_cols]
            Y   = data[self._labels+['asset_no','time_stamp']]
            y_pred        = self._model.predict(X)
            Y['prediction'] = list(self._model.predict(X))
            Y['actual'] = list(Y[self._labels].values)
            scores = computeMetrics(metrics, Y[['prediction','actual','time_stamp']], None, 'overall')
            return scores
        else:
            return "Exception occured...."
        
# use_cols =['POWER_MANAGEMENT_SYSTEM/DG2/DG2LUBOILPRESSURE',
#                  'POWER_MANAGEMENT_SYSTEM/DG2/DG2ENGINERPM',
#                  'POWER_MANAGEMENT_SYSTEM/DG2/DG2COOLANTTEMPERATURE',
#                 #  'POWER_MANAGEMENT_SYSTEM/DG2/DG2COOLANTPRESSURE',
#                  'POWER_MANAGEMENT_SYSTEM/MSB/DG2_ACTIVE_POWER',
#                  'POWER_MANAGEMENT_SYSTEM/DG2/DG2LUBOILTEMPERATURE']
# labels=['POWER_MANAGEMENT_SYSTEM/DG2/DG2LUBOILTEMPERATURE']
# window_size=1000
# stride=1
# bw=Model(use_cols,labels,window_size,stride)
# df=pd.read_csv('/datadrive2/ml_model_deployment/OPENTSDB_DATA_PUSH/lubrication_system_failure_data.csv') 
# bw.train(df.copy())
# metrics = ['r2_score','mean_absolute_error','mean_squared_error']
# print (bw.test(df.copy(),metrics))
# target_label = {'DERIVED_POWER_MANAGEMENT_SYSTEM/DG1/DG1LUBOILTEMPERATURE':{'actual_label':'POWER_MANAGEMENT_SYSTEM/DG1/DG1LUBOILTEMPERATURE','encoding':{},'operating_characteristic_type_name':'measurement',"event":"Alarm"},'DERIVED_DG1_LUBRICATION_SYSTEM_FAILURE':{'actual_label':'','encoding':{0:'Normal',1:'Warning'},'operating_characteristic_type_name':'event',"event":"Alarm"}}
# output = bw.predict(df.copy(), target_label)
# print (output.columns)
# print (output.head())
# output.to_csv('delete.csv')
# df = df.rename(columns = {'POWER_MANAGEMENT_SYSTEM/DG2/DG2LUBOILPRESSURE':'POWER_MANAGEMENT_SYSTEM/DG1/DG1LUBOILPRESSURE',
#                  'POWER_MANAGEMENT_SYSTEM/DG2/DG2ENGINERPM':'POWER_MANAGEMENT_SYSTEM/DG1/DG1ENGINERPM',
#                  'POWER_MANAGEMENT_SYSTEM/DG2/DG2COOLANTTEMPERATURE':'POWER_MANAGEMENT_SYSTEM/DG1/DG1COOLANTTEMPERATURE',
#                 #  'POWER_MANAGEMENT_SYSTEM/DG2/DG2COOLANTPRESSURE',
#                  'POWER_MANAGEMENT_SYSTEM/MSB/DG2_ACTIVE_POWER':'POWER_MANAGEMENT_SYSTEM/MSB/DG1_ACTIVE_POWER',
#                  'POWER_MANAGEMENT_SYSTEM/DG2/DG2LUBOILTEMPERATURE':'POWER_MANAGEMENT_SYSTEM/DG1/DG1LUBOILTEMPERATURE'})
# target_label = {'DERIVED_POWER_MANAGEMENT_SYSTEM/DG1/DG1LUBOILTEMPERATURE':{'actual_label':'POWER_MANAGEMENT_SYSTEM/DG1/DG1LUBOILTEMPERATURE','encoding':{},'operating_characteristic_type_name':'measurement',"event":"Alarm"},'DERIVED_DG1_LUBRICATION_SYSTEM_FAILURE':{'actual_label':'','encoding':{0:'Normal',1:'Warning'},'operating_characteristic_type_name':'event',"event":"Alarm"}}
# bw._asset = 'DG1'
# bw._labels = ['POWER_MANAGEMENT_SYSTEM/DG1/DG1LUBOILTEMPERATURE']
# bw._use_cols = ['POWER_MANAGEMENT_SYSTEM/DG1/DG1LUBOILPRESSURE',
#                  'POWER_MANAGEMENT_SYSTEM/DG1/DG1ENGINERPM',
#                  'POWER_MANAGEMENT_SYSTEM/DG1/DG1COOLANTTEMPERATURE',
#                  'POWER_MANAGEMENT_SYSTEM/MSB/DG1_ACTIVE_POWER',
#                  'POWER_MANAGEMENT_SYSTEM/DG1/DG1LUBOILTEMPERATURE']
# print (bw.predict(df.copy(), target_label))
# print (bw._hyper_parm_list)
# print (bw._train_report)
# print (bw._test_report)
