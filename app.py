import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    occupation = request.form.get("OCCUPATION")
    if_occupation_hazardous = request.form.get("IF_OCCUPATION_HAZARDOUS")
    gender = request.form.get("GENDER")
    age = float(request.form.get("AGE"))

    text_digit_vals = {'MEDICALTUTOR': 125, 'SELF-EMPLOYED': 1, 'EMPLOYED': 2,'STUDENT':3,'SHIPREPAIROFFICER':4}
    occupation = text_digit_vals[occupation]
    if_occupation_hazardous = 1 if if_occupation_hazardous == 'YES' else 0
    gender = 1 if gender == 'MALE' else 0

    features = [np.array([occupation, if_occupation_hazardous, gender, age])]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="The Proposal Score is {}".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)