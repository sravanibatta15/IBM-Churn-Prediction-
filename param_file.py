import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('par')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def check(train_ind,train_dep):
    try:
        reg=LogisticRegression()
        reg.fit(train_ind,train_dep)
        logger.info(f'Test accuracy:{accuracy_score(train_dep,reg.predict(train_ind))}')
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
            'class_weight': [None, 'balanced'],
            'max_iter': [100, 200, 500, 1000],
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'n_jobs': [None, -1],
            'l1_ratio': [None, 0.0, 0.25, 0.5, 0.75, 1.0]  # only used if penalty='elasticnet'
        }
        grid = GridSearchCV(reg, param_grid, cv=5, scoring='accuracy')
        grid.fit(train_ind,train_dep)
        logger.info(f'Best parameters:{grid.best_params_}')
        logger.info(f'Best score:{grid.best_score_}')
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')