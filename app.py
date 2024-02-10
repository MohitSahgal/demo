from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Teams and cities data
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']
cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 'Christchurch', 'Trinidad']

@app.route('/')
def index():
    return render_template('index.html', teams=teams, cities=cities)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve user input from the form
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        city = request.form['city']
        current_score = float(request.form['current_score'])
        overs = float(request.form['overs'])
        wickets = float(request.form['wickets'])
        last_five = float(request.form['last_five'])

        # Calculate additional features
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs

        # Create input DataFrame
        input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team],
                                 'city': [city], 'current_score': [current_score],
                                 'balls_left': [balls_left], 'wickets_left': [wickets_left],
                                 'crr': [crr], 'last_five': [last_five]})

        # Make prediction
        result = pipe.predict(input_df)

        # Display the predicted score
        return render_template('result.html', result=int(result[0]))

if __name__ == '__main__':
    app.run(debug=True)
