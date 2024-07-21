from flask import Flask, render_template, request
import joblib
import numpy as np

model = joblib.load(open('app/model.joblib', 'rb'))

class_names = np.array(['Bottom 6','Mid-Table', 'European Places'])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
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
    result = model.predict([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12]])
    return render_template('home.html', **locals())

if __name__ == "__main__":
    app.run(debug=True)