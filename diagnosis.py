from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    Loded_model = pickle.load(open('E:/Diabetes Prediction/trained_model.sav', 'rb'))

    # Get user input
    input_data = [request.form[f] for f in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

    data_np_array = np.asarray(input_data)
    data_shape = data_np_array.reshape(1,-1)
    pred = Loded_model.predict(data_shape)

    if pred[0] == 0:
        diagnosis = 'The person is not diabetic'
    else:
        diagnosis = 'The person is diabetic'

    return render_template('result.html', diagnosis=diagnosis)

if __name__ == '__main__':
    app.run(debug=True)
