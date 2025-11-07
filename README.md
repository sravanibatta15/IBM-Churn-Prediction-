# 🧠 AI-Powered Customer Retention Prediction System

This project — **AI-Powered Customer Retention Prediction System** — is a machine-learning-driven solution designed to predict **customer churn** in the telecom sector.  
It helps businesses identify customers who are likely to leave and take proactive actions to retain them.

---

## 🚀 Project Overview

Customer churn has a direct impact on business revenue, brand reputation, and growth.  
This project uses **AI & Data Science techniques** to build a predictive system that analyzes customer data (demographics, billing, services, usage patterns) to anticipate churn behavior.

The workflow includes:
- Data preprocessing & feature engineering  
- Handling missing values & outliers  
- Encoding categorical variables  
- Balancing the dataset with **SMOTE**  
- Feature scaling using **StandardScaler**  
- Model training with 8 algorithms  
- Evaluation with metrics like Accuracy, Precision, Recall, F1, and ROC-AUC  

The final selected model — **XGBoost** — achieved the best performance, offering reliable churn predictions for actionable business insights.

---


## 🧩 Tools & Technologies

| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Balancing | SMOTE (from imbalanced-learn) |

---

## 📊 Key Insights
- **Short-tenure and month-to-month contract** customers churn the most.  
- **High monthly charges** correlate with higher churn rates.  
- **Auto-pay and long-term contracts** reduce churn.  
- Customers with **add-on services** (Tech Support, Online Security) show higher retention.  
- **Electronic check** payment users churn the most — highlighting convenience as a retention factor.  
- **XGBoost** outperformed other models in accuracy and generalization.

---

## 🧠 Model Comparison Summary

| Algorithm | Accuracy | F1-Score | AUC | Remarks |
|------------|-----------|----------|-----|----------|
| KNN | Moderate | Low | – | Distance-based, sensitive to scaling |
| Naive Bayes | Moderate | Low | – | Assumes independence |
| Logistic Regression | 0.757 | Moderate | – | Interpretable baseline |
| Decision Tree | Moderate | – | – | Prone to overfitting |
| Random Forest | 0.758 | High | – | Strong ensemble |
| SVM | Moderate | – | – | Sensitive to imbalance |
| Gradient Boosting | High | 0.60 | 0.72 | Balanced performance |
| **XGBoost (Final)** | **0.797** | **0.60** | **0.729** | **Best performer** |

---

## 🏁 Final Outcome

✅ **XGBoost** was chosen as the final model for deployment.  
✅ The model accurately identifies at-risk customers, enabling targeted retention campaigns.  
✅ Businesses can use this system to **reduce churn, improve satisfaction**, and **increase long-term revenue**.

---

## 🔮 Future Scope
- Develop a **real-time churn prediction API** for integration with CRM dashboards.  
- Add **deep-learning-based models** for advanced accuracy.  
- Include **customer feedback sentiment** to enrich predictive signals.  
- Deploy on **cloud platforms** for scalable inference.

---

## 📞 Contact & Support
If you find this project helpful or want to collaborate, feel free to reach out:

**👤 Author:** *Batta Siva Sai Sravani*  
**📧 Email:** [sivasaisravani@gmail.com]  
**📱 Phone:** [8639868362]  
**🔗 LinkedIn:** [https://www.linkedin.com/in/siva-sai-sravani-007772286/]
** For Project Here:** [https://ibm-churn-prediction.onrender.com]

---

### ⭐ Don’t forget to give this repository a star if you like it!
