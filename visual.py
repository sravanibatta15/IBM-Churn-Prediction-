import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import log_code
logger = log_code.setup_logging('visual')
import sys
class Visual:
    def plot_transformations(all_transformed, method):
        try:
            for i in all_transformed.columns:
                if method in i:
                    logger.info(i)
                    plt.figure(figsize=(10,4))
                    plt.subplot(1, 3, 1)
                    all_transformed[i].plot(kind='kde', color='r', label='Check')
                    plt.title(f"{i}KDE{method}")
                    plt.subplot(1, 3, 2)
                    sns.boxplot(all_transformed[i].dropna(), label='check')
                    plt.title(f"{i}Boxplot{method}")
                    plt.subplot(1, 3, 3)
                    plt.title(f"{i}Probability Plot{method}")
                    stats.probplot(all_transformed[i], dist='norm', plot=plt)
                    plt.tight_layout()
                    plt.show()
        except Exception as e:
            e_type, e_msg, e_linno = sys.exc_info()
            logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')