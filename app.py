# Importing necessary dependencies
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

# Initializing a Flask app
app = Flask(__name__)

# Load the model from .pickle file
model_path = r"C:/Users/HP/Desktop/Prndrive/Loan Aproval Prediction/Flash/c_card aproval pred.pickle"
with open(model_path, 'rb') as handle:
    model = pickle.load(handle)

# Route to display the home page
@app.route('/')
def home():
    return render_template('ccaindex.html')

# Route to render prediction input form
@app.route('/Prediction', methods=['POST', 'GET'])
def prediction():
    return render_template('ccaindex1.html')

# Route to go back to home
@app.route('/Home', methods=['POST', 'GET'])
def my_home():
    return render_template('ccaindex.html')

# Route to handle prediction
@app.route('/predict', methods=["POST", "GET"])
def predict():
    # Read inputs given by the user
    input_features = [float(x) for x in request.form.values()]
    feature_values = [np.array(input_features)]

    feature_names = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                     'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                     'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
                     'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'paid_off',
                     '#_of_pastdues', 'no_Loan']

    x = pd.DataFrame(feature_values, columns=feature_names)

    # Predict using loaded model
    pred = model.predict(x)
    print(pred)

    # Interpret prediction
    if pred[0] == 1:
        prediction = "Eligible"
    else:
        prediction = "Not Eligible"

    # Show prediction result in UI
    return render_template("Results.html", prediction=prediction)

# Start the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=False, use_reloader=False)
