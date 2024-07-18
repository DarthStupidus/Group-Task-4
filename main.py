from flask import Flask, render_template, request
import joblib
import numpy as np
import os

model = joblib.load('app/model.joblib')

class_names = np.array(['Bottom 6','Mid-Table', 'European Places'])

app = Flask(__name__)

@app.get('/')
def reed_root():
    return render_template('home.html')

@app.post('/predict')
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    data12 = request.form['l']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))