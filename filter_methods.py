import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import sys
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency,pearsonr
from log_code import setup_logging
logger = setup_logging('fill')

def chi_square(df,df1,y):
    try:
        logger.info(f'faff:{df.isnull().sum()}')
        selector = SelectKBest(score_func=chi2, k=2)
        x_new = selector.fit_transform(df, y)
        # Score card creation
        chi2_score = pd.DataFrame(
            {
                'Features': df.columns,
                'Chi_score': selector.scores_,
                'p values': selector.pvalues_
            }
        ).sort_values(by='Chi_score', ascending=False)
        logger.info(f'Chi score data frame :\n {chi2_score}')
        chi2_score = chi2_score[chi2_score['Features'] != 'sim']
        remove_features = chi2_score[chi2_score['p values'] > 0.05]['Features']
        df_filtered = df.drop(columns=remove_features)
        df1_filtered=df1.drop(columns=remove_features)
        logger.info(f"Removed columns: {list(remove_features)}")
        return  df_filtered,df1_filtered
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

def anova(a,c,b):
    try:
        selector1=SelectKBest(score_func=f_classif,k=2)
        y_new=selector1.fit_transform(a,b)
        anova_score=pd.DataFrame({
                    'Features': a.columns,
                    'anova_score': selector1.scores_,
                    'p values': selector1.pvalues_}
                    ).sort_values(by='anova_score',ascending=False)
        logger.info(f'Anova score data frame:\n{anova_score}')
        remove_feature = anova_score[anova_score['p values'] > 0.05]['Features']
        df_filter = a.drop(columns=remove_feature)
        df1_filter = c.drop(columns=remove_feature)
        logger.info(f"Removed columns anova: {list(remove_feature)}")
        return df_filter,df1_filter
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

def t_test(p,q,r):
    try:
        ttest_results = []
        for i in p.columns:
            group1 = p[r == 0][i]
            group2 = p[r == 1][i]
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
            ttest_results.append({'Features': i, 't_stat': t_stat, 'p_value': p_val})
        ttest_df = pd.DataFrame(ttest_results).sort_values(by='t_stat', ascending=False)
        logger.info(f"T-test results:\n{ttest_df}")
        remove_feature1 = ttest_df[ttest_df['p_value'] > 0.05]['Features']
        df_fil = p.drop(columns=remove_feature1)
        df1_fil1= q.drop(columns=remove_feature1)
        logger.info(f"Removed columns t-test: {list(remove_feature1)}")
        return df_fil, df1_fil1
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')

def correlation(c,c1):
    try:
        feature=[]
        for i in c.columns:
            r, p = pearsonr(c[i],c1)
            logger.info(f'{i}----->{r}')
            if p < 0.05:
                feature.append(i)
            logger.info(f'{feature}')
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')