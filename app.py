from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# load model & scaler
model = pickle.load(open("grad_boost.pkl", "rb"))
scaler = pickle.load(open("stand_scalar.pkl", "rb"))

final_cols = model.feature_names_in_

CONTRACT_MAP = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
SIM_MAP = {'Jio': 0, 'Airtel': 1, 'Vi': 2, 'BSNL': 3}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        f = request.form

        data = {
            'gender'          : f.get('gender', 'Male'),
            'Partner'         : f.get('Partner', 'No'),
            'Dependents'      : f.get('Dependents', 'No'),
            'PhoneService'    : f.get('PhoneService', 'No'),
            'MultipleLines'   : f.get('MultipleLines', 'No'),
            'InternetService' : f.get('InternetService', 'No'),
            'OnlineSecurity'  : f.get('OnlineSecurity', 'No'),
            'OnlineBackup'    : f.get('OnlineBackup', 'No'),
            'DeviceProtection': f.get('DeviceProtection', 'No'),
            'TechSupport'     : f.get('TechSupport', 'No'),
            'StreamingTV'     : f.get('StreamingTV', 'No'),
            'StreamingMovies' : f.get('StreamingMovies', 'No'),
            'PaperlessBilling': f.get('PaperlessBilling', 'No'),
            'PaymentMethod'   : f.get('PaymentMethod', 'Electronic check'),
            'Contract'        : f.get('Contract', 'Month-to-month'),
            'sim'             : f.get('sim', 'Jio'),
        }

        data['SeniorCitizen'] = int(f.get('SeniorCitizen', 0))
        data['JoinYear']      = int(f.get('JoinYear', 2020))

        m_raw = float(f.get('MonthlyCharges', 0))
        t_raw = float(f.get('TotalCharges', 0))
        data['MonthlyCharges_qan_quantiles'] = m_raw
        data['TotalCharges_KNN_imp_qan_quantiles'] = t_raw

        df = pd.DataFrame([data])

        cat_cols = [
            'gender','Partner','Dependents','PhoneService','MultipleLines',
            'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','PaymentMethod'
        ]
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        df['Contract_con'] = CONTRACT_MAP.get(f.get('Contract', 'Month-to-month'), 0)
        df.drop(columns=['Contract'], inplace=True)
        df['sim'] = SIM_MAP.get(f.get('sim', 'Jio'), 0)

        sc_cols = ['MonthlyCharges_qan_quantiles','TotalCharges_KNN_imp_qan_quantiles']
        df[sc_cols] = scaler.transform(df[sc_cols])

        for c in final_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[final_cols]

        pred = model.predict(df)[0]
        msg = "✅ Customer Will Stay" if pred == 0 else "⚠️ Customer Likely to Churn"

        return render_template("index.html", prediction_text=msg, selected_sim=f.get('sim','Jio'))
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}", selected_sim=f.get('sim','Jio'))


@app.route("/about_developer")
def about_developer():
    return render_template("about_developer.html")


@app.route("/about_model")
def about_model():
    return render_template("about_model.html")


if __name__ == "__main__":
    app.run(debug=True)
