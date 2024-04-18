import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open("majorproject/lrmodel.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
       
        protein = float(request.form['protein'])
        fat = float(request.form['fat'])
        vitaminC = float(request.form['vitaminC'])
        fibre = float(request.form['fibre'])
        input_features = np.array([[protein, fat, vitaminC, fibre]])
        cluster_prediction = model.predict(input_features)
        return render_template('result.html', prediction=cluster_prediction[0])

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
