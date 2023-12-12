from flask import Flask, render_template, request
import numpy as np
import pickle
app = Flask(__name__)
with open('best_models.pkl','rb')as file:
    model=pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit',methods=['POST'])
def submit():
    temperature=float(request.form['Temperature[c]'])
    humidity=float(request.form['Humidity[%]'])
    tvoc= float(request.form['TVOC[ppb]'])
    raw_h2=float(request.form['Raw H2'])
    raw_ethanol=float(request.form['Raw Ethanol'])
    pressure=float(request.form['Pressure[hpa]'])
    nc0_5=float(request.form['NC0.5'])
    cnt=float(request.form['CNT'])
    final_features=np.array([[temperature,humidity,tvoc,raw_h2,raw_ethanol,pressure,nc0_5,cnt]])
    prediction=model.predict(final_features)[0]

    if prediction== 0:
        prediction_text='The input does not indicate smoke detection.'
    else:
       prediction_text='The input indicates smoke detection.'
    return render_template('submit.html',prediction_text=prediction_text)   
if __name__ == '__main__':
   app.run(debug=True)    