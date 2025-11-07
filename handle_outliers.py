import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import sys
import feature_engine
from feature_engine.outliers import Winsorizer
from sklearn.ensemble import IsolationForest
from log_code import setup_logging
logger = setup_logging('handling_outlier')

def caping(x_train, x_test,method,fold):
    try:
        df = x_train.copy()
        df1 = x_test.copy()
        cols = ['MonthlyCharges_qan', 'TotalCharges_KNN_imp_qan']
        winsor = Winsorizer(capping_method=method, tail='both', fold=fold, variables=cols)
        df_trans = winsor.fit_transform(df)
        df1_trans = winsor.transform(df1)
        for i in cols:
            new_col_train = f"{i}_{method}"
            new_col_test = f"{i}_{method}"
            df[new_col_train] = df_trans[i]
            df1[new_col_test] = df1_trans[i]

            plt.figure(figsize=(10,3))
            plt.subplot(1,3,1)
            sns.boxplot(x=df[new_col_train], color='r')
            plt.title(f'{new_col_train} - Box {method} Capping')

            plt.subplot(1, 3, 2)
            df[new_col_train].plot(kind='kde',color='skyblue')
            plt.title(f'{new_col_train} - Bell {method} Capping')
            plt.xlabel(new_col_train)
            plt.ylabel('Density')

            plt.subplot(1, 3, 3)
            stats.probplot(df[new_col_train], dist="norm", plot=plt)
            plt.title(f'{new_col_train} - Q-Q Plot')
            plt.show()
            # plt.figure(figsize=(8, 3))
            # sns.boxplot(x=df1[new_col_test], color='b')
            # plt.title(f'{new_col_test} - After {method.upper()} Capping')
            # plt.show()
        logger.info(f"Capping using {method.upper()} Winsorizer completed successfully.")
        logger.info(f'train:{df.columns}')
        logger.info(f'test:{df1.columns}')
        return df, df1
    except Exception:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f"Issue is at line {e_linno.tb_lineno} due to {e_msg}")

# def Outlier_trim(self): Data is lossing
#     try:
#         log.info("Outlier_trim has started................")
#         tf1 = OutlierTrimmer()
#     except Exception:
#         exc_type, exc_msg, exc_tb = sys.exc_info()
#         log.error(f'{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}')

# def Isolation_forests(x_train):
#     try:
#         logger.info("IsolationForest with capping has started......")
#         cols = ['MonthlyCharges_qan', 'TotalCharges_KNN_imp_qan']
#         iso = IsolationForest(contamination=0.1, random_state=42)
#         cols['Outlier'] = iso.fit_predict(cols)
#         for i in df_num.columns:
#             if i != 'Outlier':
#                 Q1 = cols[i].quantile(0.25)
#                 Q3 = cols[i].quantile(0.75)
#                 IQR = Q3 - Q1
#                 lower_limit = Q1 - 1.5 * IQR
#                 upper_limit = Q3 + 1.5 * IQR
#                 for idx in cols[cols['Outlier'] == -1].index:
#                     value = cols.loc[idx,i]
#                     # log.info(value)
#                     # If value < lower limit → set to lower limit
#                     if value < lower_limit:
#                         cols.loc[idx,i] = lower_limit
#                     # If value > upper limit → set to upper limit
#                     elif value > upper_limit:
#                         cols.loc[idx,i] = upper_limit
#                     # else: keep as is
#         cols=cols.drop(columns=['Outlier'])
#         x_train[cols.columns]=x_train[cols.columns]
#         logger.info(f'After checking the shape:{df.shape}')
#         logger.info("IsolationForest capping completed successfully.")
#         return x_train
#     except Exception:
#         e_type, e_msg, e_linno = sys.exc_info()
#         logger.info(f"Issue is at line {e_linno.tb_lineno} due to {e_msg}")


