from scipy import stats
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
import os
import sys
from log_code import setup_logging
logger=setup_logging('main')
from variable_transformation import quantile
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from knn_imputer import knn_imp
#from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from variable_transformation import quantile
from visual import Visual
import seaborn as sns
from handle_outliers import caping
from filter_methods import chi_square, anova,t_test,correlation
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from train_alogs import common
from param_file import check
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

class CHURN:
    def __init__(self,path):
        try:
            self.df=pd.read_csv(path)
            self.df=self.df.drop(['customerID','charges','gateway','tax'],axis=1)
            self.df['TotalCharges']=pd.to_numeric(self.df['TotalCharges'],errors='coerce')
            logger.info(f'Data loaded successfully:{self.df.shape}')
            logger.info(self.df.head(5))
            self.x=self.df.drop(['Churn'],axis=1)#independent
            self.y=self.df['Churn']#dependent
            self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.x,self.y,test_size=0.2,random_state=24)
            logger.info(f'training data shape:{self.x_train.shape}')
            logger.info(f'testing data shape:{self.x_test.shape}')
            logger.info(f'Missing values in the data:{self.df.isnull().sum()}')
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')
    def knn(self):
        try:
            logger.info(f'Before:{self.x_train.columns}')
            before_imp = self.x_train['TotalCharges'].copy()
            res1, res2 = knn_imp(self.x_train, self.x_test)
            logger.info(f'Before{self.x_train.columns}')
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            before_imp.hist(bins=30, color='r')
            plt.title("Before Imputation")
            plt.subplot(1, 2, 2)
            self.x_train['TotalCharges_KNN_imp'].hist(bins=30, color='blue')
            plt.title("After Imputation")
            plt.tight_layout()
            plt.show()
            self.x_train=self.x_train.drop(['tenure'],axis=1)
            self.x_test=self.x_test.drop(['tenure'],axis=1)
            self.x_train_num1=self.x_train.select_dtypes(exclude='object')
            self.x_train_cat1=self.x_train.select_dtypes(include='object')
            logger.info(f'tssf:{self.x_train_num1.columns}')
            logger.info(f'vwcd:{self.x_train_cat1.columns}')
            self.x_test_num1 = self.x_test.select_dtypes(exclude='object')
            logger.info(f'safb:{type(self.x_test_num1)}')
            self.x_test_cat1 = self.x_test.select_dtypes(include='object')
            logger.info(f'tsasf:{self.x_test_num1.columns}')
            logger.info(f'vsaccd:{self.x_test_cat1.columns}')
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

    def variable_transform(self):
        try:
            cols=['MonthlyCharges','TotalCharges_KNN_imp']
            self.dummy = self.x_train_num1[cols].copy()
            self.dumm = self.x_test_num1[cols].copy()
            logger.info(f'dummy check{self.dummy.columns}')
            logger.info(f'dumm check{self.dumm.columns}')
            all1,all2=quantile(self.dummy,self.dumm)
            #all1.index=self.x_train_num.index
            #all2.index=self.x_test_num.index
            for i in all1.columns:
                self.x_train_num1[i]=all1[i]
            for j in all2.columns:
                self.x_test_num1[j]=all2[j]
            logger.info(f'train:{self.x_train_num1.columns}')
            logger.info(type(self.x_test_num1))
            logger.info(f'test:{self.x_test_num1.columns}')
            Visual.plot_transformations(all1,'_qan')
            Visual.plot_transformations(all2,'_qan')
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')
    def outliers(self):
        try:
            # a1,b1=caping(self.x_train_num1,self.x_test_num1,method='iqr',fold=1.5)
            # a2,b2=caping(self.x_train_num1,self.x_test_num1,method='gaussian',fold=2.5)
            # a3,b3=caping(self.x_train_num1,self.x_test_num1, method='mad', fold=1.5)
            a4,b4=caping(self.x_train_num1,self.x_test_num1, method='quantiles',fold=0.01)
            # a5=Isolation_forests(self.x_train_num1)
            for i in a4.columns:
                self.x_train_num1[i]=a4[i]
            for j in b4.columns:
                self.x_test_num1[j]=b4[j]
            logger.info(f'train:{self.x_train_num1.columns}')
            logger.info(f'test:{self.x_test_num1.columns}')
            Visual.plot_transformations(self.x_test_num1, '_quantiles')
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

    def encoding(self):
        try:
            cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']
            #one hot encoding
            oh = OneHotEncoder(categories='auto', drop='first', handle_unknown='ignore')
            oh.fit(self.x_train_cat1[cols])
            logger.info(f'{oh.categories_}')
            logger.info(f'{oh.get_feature_names_out()}')
            res = oh.transform(self.x_train_cat1[cols]).toarray()
            res1 = oh.transform(self.x_test_cat1[cols]).toarray()
            f = pd.DataFrame(res, columns=oh.get_feature_names_out(), index=self.x_train_cat1.index)
            f1 = pd.DataFrame(res1, columns=oh.get_feature_names_out(), index=self.x_test_cat1.index)
            self.x_train_cat1 = pd.concat([self.x_train_cat1.drop(columns=cols), f], axis=1)
            self.x_test_cat1 = pd.concat([self.x_test_cat1.drop(columns=cols), f1], axis=1)
            logger.info(self.x_train_cat1.isnull().sum())
            logger.info(self.x_test_cat1.isnull().sum())

            #odinal encoding
            ob = OrdinalEncoder()
            ob.fit(self.x_train_cat1[['Contract']])
            logger.info(f'{ob.categories_}')
            logger.info(f'column names:{ob.get_feature_names_out()}')
            res2 = ob.transform(self.x_train_cat1[['Contract']])
            res2_test = ob.transform(self.x_test_cat1[['Contract']])
            c_names = ob.get_feature_names_out()
            f2 = pd.DataFrame(res2, columns=c_names + ['_con'], index=self.x_train_cat1.index)
            f2_test = pd.DataFrame(res2_test, columns=c_names + ['_con'], index=self.x_test_cat1.index)
            self.x_train_cat1 = pd.concat([self.x_train_cat1, f2], axis=1)
            self.x_test_cat1 = pd.concat([self.x_test_cat1, f2_test], axis=1)
            self.x_train_cat1 = self.x_train_cat1.drop(['Contract'],axis=1)
            self.x_test_cat1 = self.x_test_cat1.drop(['Contract'],axis=1)
            logger.info(f'{self.x_train_cat1.columns}')
            logger.info(f'{self.x_train_cat1.sample(5)}')
            logger.info(f'{self.x_train_cat1.isnull().sum()}')

            #label encoding
            logger.info(f'dependent:{self.y_train[:10]}')
            lb=LabelEncoder()
            lb.fit(self.y_train)
            self.y_train=lb.transform(self.y_train)
            self.y_test=lb.transform(self.y_test)
            logger.info(f'detailed:{lb.classes_}')
            logger.info(f'dependentzzz:{self.y_train[:10]}')
            logger.info(f'y_train_data:{self.y_train.shape}')
            logger.info(f'y_test_data:{self.y_test.shape}')

            # 0 -> No
            # 1 -> Yes
            logger.info(f'Check null1 in before the drop {self.x_train_num1["SeniorCitizen"]}')
            self.x_train_cat1['SeniorCitizen']=self.x_train_num1['SeniorCitizen']
            self.x_test_cat1['SeniorCitizen']=self.x_test_num1['SeniorCitizen']
            self.x_train_cat1['sim'] = self.x_train_cat1['sim'].map({'Jio': 0, 'Airtel': 1, 'Vi': 2, 'BSNL': 3})
            self.x_test_cat1['sim'] = self.x_test_cat1['sim'].map({'Jio': 0, 'Airtel': 1, 'Vi': 2, 'BSNL': 3})
            logger.info(f'encode:{self.x_train_cat1}')
            self.x_train_num1=self.x_train_num1.drop(['SeniorCitizen'],axis=1)
            self.x_test_num1=self.x_test_num1.drop(['SeniorCitizen'],axis=1)
            logger.info(f'check the null in the data frame:{self.x_train_cat1.isnull().sum()}')
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

    def filter(self):
        try:
            ch,ch1=chi_square(self.x_train_cat1,self.x_test_cat1,self.y_train)
            self.x_train_cat11=ch
            self.x_test_cat11=ch1
            logger.info(f'After chi2 test checking the shape and columns{ch.shape}\n{ch.columns}')
            logger.info(f'________________________')
            cols_remove_unwanted = ['MonthlyCharges_qan','TotalCharges_KNN_imp_qan','TotalCharges_KNN_imp','TotalCharges','MonthlyCharges']
            self.x_train_num1 = self.x_train_num1.drop(cols_remove_unwanted, axis=1)
            self.x_test_num1 = self.x_test_num1.drop(cols_remove_unwanted, axis=1)
            logger.info(f'{self.x_train_num1.isnull().sum()}')
            av,av1=anova(self.x_train_num1,self.x_test_num1,self.y_train)
            tt,tt1=t_test(self.x_train_num1,self.x_test_num1,self.y_train)
            cc=correlation(self.x_train_num1,self.y_train)
            logger.info(f'After filtering:{self.x_train_cat11.columns}')
            logger.info(f'numerical:{self.x_train_num1.shape}')
            logger.info(f'numerical:{self.x_train_cat11.shape}')
            logger.info(f'numerical:{self.x_test_num1.shape}')
            logger.info(f'numerical:{self.x_test_cat11.shape}')
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

    def merge_data(self):
        try:
            self.x_train_num1.reset_index(drop=True,inplace=True)
            self.x_train_cat11.reset_index(drop=True,inplace=True)
            self.x_test_num1.reset_index(drop=True,inplace=True)
            self.x_test_cat11.reset_index(drop=True,inplace=True)
            self.training_data=pd.concat([self.x_train_num1,self.x_train_cat11],axis=1)
            self.testing_data=pd.concat([self.x_test_num1,self.x_test_cat11],axis=1)
            logger.info(f'Training_data shape : {self.training_data.shape} -> {self.training_data.columns}')
            logger.info(f'Testing_data shape : {self.testing_data.shape} -> {self.testing_data.columns}')
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')
    def balanced_data(self):
        try:
            logger.info(f'_________Before Balancing__________')
            logger.info(f'Total row for Good category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 1)}')
            logger.info(f'Total row for Bad category in training data {self.training_data.shape[0]} was : {sum(self.y_train == 0)}')
            logger.info(f'---------------After Balancing-------------------------')
            sm = SMOTE(random_state=42)
            self.training_data_res, self.y_train_res = sm.fit_resample(self.training_data, self.y_train)
            logger.info(f'Total row for Good category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 1)}')
            logger.info(f'Total row for Bad category in training data {self.training_data_res.shape[0]} was : {sum(self.y_train_res == 0)}')
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

    def feature_scaling(self):
        try:
            backup_year_train = self.training_data_res['JoinYear'].copy()
            backup_year_test = self.testing_data['JoinYear'].copy()
            self.training_data_res.drop(['JoinYear'], axis=1, inplace=True)
            self.testing_data.drop(['JoinYear'], axis=1, inplace=True)
            scale_cols = ['MonthlyCharges_qan_quantiles', 'TotalCharges_KNN_imp_qan_quantiles']
            self.ms = StandardScaler()
            self.ms.fit(self.training_data_res[scale_cols])
            scaled_train = pd.DataFrame(self.ms.transform(self.training_data_res[scale_cols]),columns=scale_cols, index=self.training_data_res.index)
            scaled_test = pd.DataFrame(self.ms.transform(self.testing_data[scale_cols]),columns=scale_cols, index=self.testing_data.index)
            other_train = self.training_data_res.drop(scale_cols, axis=1, errors='ignore')
            other_test = self.testing_data.drop(scale_cols, axis=1, errors='ignore')
            self.training_data_res_t = pd.concat([other_train, scaled_train], axis=1)
            self.testing_data_t = pd.concat([other_test, scaled_test], axis=1)
            self.training_data_res_t['JoinYear'] = backup_year_train
            self.testing_data_t['JoinYear'] = backup_year_test
            logger.info(self.testing_data_t)
            #model = common(self.training_data_t, self.y_train_res, self.testing_data_t, self.y_test)
            logger.info("Scaling applied on MonthlyCharges_qan_quantiles & TotalCharges_KNN_imp_qan_quantiles.")
            with open(r"stand_scalar.pkl", "wb") as f:
                pickle.dump(self.ms, f)
            #self.training_data_t.to_csv('./Data/final.csv', index=False)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f"Issue is : {er_lin.tb_lineno} : due to : {er_msg}")

    def train_models(self):
        try:
            common(self.training_data_res_t, self.y_train_res, self.testing_data_t, self.y_test)
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

    def parameters(self):
        try:
            # self.data_ind=self.training_data_res_t.head(200)
            # self.data_dep=self.y_train_res[:200]
            # check(self.data_ind,self.data_dep)
            logger.info(f'__________Finalized Model___________')
            # self.reg1=LogisticRegression(C=1.0,class_weight=None,l1_ratio=None,max_iter=100,multi_class='auto',n_jobs=None,penalty='l1',solver='liblinear')
            # self.reg1.fit(self.training_data_res_t,self.y_train_res)
            self.reg2=GradientBoostingClassifier()
            self.reg2.fit(self.training_data_res_t,self.y_train_res)
            logger.info(f'Train accuracy:{accuracy_score(self.y_train_res,self.reg2.predict(self.training_data_res_t))}')
            logger.info(f'Test accuracy:{accuracy_score(self.y_test,self.reg2.predict(self.testing_data_t))}')
            logger.info(f'=====Model Saving======')
            with open('grad_boost.pkl', 'wb') as f:
                pickle.dump(self.reg2,f)
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

if __name__=='__main__':
    try:
        path='C:\\Users\\sravs\\Downloads\\churn_prject\\WA_Fn-UseC_-Telco-Customer-Churn.csv'
        obj=CHURN(path)
        obj.knn()
        obj.variable_transform()
        obj.outliers()
        obj.encoding()
        obj.filter()
        obj.merge_data()
        obj.balanced_data()
        obj.feature_scaling()
        obj.train_models()
        obj.parameters()
    except Exception as e:
            e_type,e_msg,e_linno=sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')