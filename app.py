from flask import Flask, render_template, request, url_for, jsonify, redirect
import pandas as pd
from matplotlib import pyplot as plt
import base64, shutil, os, pickle
from io import BytesIO
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
import  numpy as np
from datetime import datetime
month_to_day = {"1":0, "2":31, "3":59, "4":90, "5":120, "6":151, "7":181, "8":212, "9":243, "10":273, "11":304, "12":334}
def createapp():
    app = Flask(__name__)
    return app
app = createapp()

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = \
                                      'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class TECParams(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    day_of_year = db.Column(db.Integer, nullable=False)
    hour_of_day = db.Column(db.Integer, nullable=False)
    rz_12 = db.Column(db.Integer, nullable=False)
    ig_12 = db.Column(db.Integer, nullable=False)
    ap_index = db.Column(db.Float, nullable=False)
    kp_index = db.Column(db.Float, nullable=False)
    tec_output = db.Column(db.Float, nullable=True)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"


@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get the form data
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Create a new user object
        new_user = User(username=username, email=email, password=password)

        # Add the user to the database
        db.session.add(new_user)
        db.session.commit()

        # Redirect to the login page
        return redirect(url_for('login'))
    else:
        return render_template('register.html')

@app.route('/signin', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get the form data
        email = request.form['email']
        password = request.form['password']

        # Check if the user exists in the database
        user = User.query.filter_by(email=email).first()

        if user and user.password == password:
            # Redirect to the home page or a protected resource
            return redirect(url_for('index'))
        else:
            # Show an error message
            return render_template('signin.html', error='Invalid email or password')
    else:
        return render_template('signin.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        year = request.form['year']
        day_of_year = month_to_day[(request.form['month'])] + int(request.form['day'])
        hour_of_day = int(request.form['hour_of_day'])
        rz_12 = request.form['rz_12']
        ig_12 = request.form['ig_12']
        ap_index = request.form['ap_index']
        kp_index = request.form['kp_index']
        with open("TEC_model.pkl", "rb") as file:
            current_model = pickle.load(file)
        inputs = np.array([[year, day_of_year, hour_of_day, rz_12, ig_12, ap_index, kp_index]])
        inputs.reshape(1, -1)
        prediction = current_model.predict(inputs) # Passing in variables for prediction
        tec_output = prediction
        tec_input = TECParams(year=year,
                                   day_of_year=day_of_year,
                                   hour_of_day=hour_of_day,
                                   rz_12=rz_12,
                                   ig_12=ig_12,
                                   ap_index=ap_index,
                                   kp_index=kp_index,
                                   tec_output=tec_output
                                     )
        db.session.add(tec_input)
        db.session.commit()
    return render_template('index.html', prediction_text='the value of TEC content is {}'.format(tec_output))

@app.route('/plot')
def plot():
    with open("TEC_model.pkl", "rb") as file:
        current_model = pickle.load(file)

    def predictor():
        prediction = current_model.predict() # Passing in variables for prediction
        return prediction
    pred_inputs = []
    prediction_outputs = []
    for i in pred_inputs:
        prediction_outputs.append(predictor(i))
    print(prediction_outputs)
    fig, ax = plt.subplots()
    prediction_outputs.plot.line()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    prediction_output =  '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta http-equiv="X-UA-Compatible" content="IE=edge"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Document</title></head><body>'+ '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '</body></html>'

    with open('test.html','w') as f:
        f.write(prediction_output)
    shutil.move(r"C:\Users\Roshan\Documents\Programcodes\python\FlaskTutorial\test.html", r"C:\Users\Roshan\Documents\Programcodes\python\FlaskTutorial\templates\test.html")
    return render_template('test.html')

if __name__ == "__main__":
    app.run(debug=True)