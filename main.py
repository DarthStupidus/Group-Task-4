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
    squad_value = request.form['squad_value']
    net_summer_transfer_spend = request.form['net_summer_transfer_spend']
    average_attendance = request.form['average_attendance']
    wage_bill = request.form['wage_bill']
    previous_season_disciplinary_points = request.form['previous_season_disciplinary_points']
    net_number_of_transfers = request.form['net_number_of_transfers']
    previous_season_goal_difference = request.form['previous_season_goal_difference']
    previous_season_position = request.form['previous_season_position']
    games_played = request.form['games_played']
    retained_manager = request.form['retained_manager']
    retained_captain = request.form['retained_captain']
    number_of_managers_employed_in_the_season = request.form['number_of_managers_employed_in_the_season']
    result = model.predict([[int(squad_value), int(net_summer_transfer_spend), bool(retained_manager), int(average_attendance), int(wage_bill), bool(retained_captain), int(previous_season_disciplinary_points), int(net_number_of_transfers), int(previous_season_goal_difference), int(previous_season_position), int(games_played), int(number_of_managers_employed_in_the_season)]])
    return render_template('home.html', **locals())

if __name__ == "__main__":
    app.run(debug=True)