import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import logging
import os
import sys
from log_code import setup_logging
logger=setup_logging('knn_imputer')
from sklearn.impute import KNNImputer

def knn_imp(x_train,x_test):
    try:
        total_charges = x_train['TotalCharges'].copy()
        print(total_charges.isnull().sum())
        total_char = x_test['TotalCharges'].copy()
        # x_train['TotalCharges']=pd.to_numeric(x_train['TotalCharges'],errors='coerce')
        # x_test['TotalCharges']=pd.to_numeric(x_test['TotalCharges'],errors='coerce')
        cont_cols = ['MonthlyCharges', 'TotalCharges']
        x_train_cont = x_train[cont_cols].copy()
        x_test_cont = x_test[cont_cols].copy()
        imp = KNNImputer(n_neighbors=10)
        x_train_imp = imp.fit_transform(x_train_cont)
        x_train_cont_imp = pd.DataFrame(x_train_imp, columns=cont_cols, index=x_train.index)
        x_test_imp = imp.transform(x_test_cont)
        x_test_cont_imp = pd.DataFrame(x_test_imp, columns=cont_cols, index=x_test.index)
        x_train['TotalCharges_KNN_imp'] = x_train_cont_imp['TotalCharges']
        x_test['TotalCharges_KNN_imp'] = x_test_cont_imp['TotalCharges']
        std_before = total_charges.std()
        std_after = x_train['TotalCharges_KNN_imp'].std()
        logger.info(f"Standard Deviation BEFORE imputation (TotalCharges): {std_before:.4f}")
        logger.info(f"Standard Deviation AFTER imputation (TotalCharges): {std_after:.4f}")
        logger.info(type(x_train))
        logger.info(type(x_test))
        logger.info(f'Knn check{x_train.columns}')
        return x_train, x_test

    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')