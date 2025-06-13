from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and preprocessing stuff
model = pickle.load(open("Insurance_prediction_model.pkl", "rb"))
ct = pickle.load(open("column_transformer.pkl", "rb"))
st = pickle.load(open("feature_scaler.pkl", "rb"))
st1 = pickle.load(open("target_scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']

    input_data = [age, sex, bmi, children, smoker, region]
    rearranged = [input_data[0], input_data[1], input_data[2], input_data[3], input_data[4], input_data[5]]

    transformed = ct.transform([rearranged])
    transformed[:, [-2, -3]] = st.transform(transformed[:, [-2, -3]])

    prediction_scaled = model.predict(transformed)
    prediction = st1.inverse_transform(prediction_scaled.reshape(-1,1))[0][0]

    return render_template("index.html", prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
