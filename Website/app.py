import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, render_template, request, redirect

# “annual_inc”, “fico_range_low”, “term”, “loan_amnt”

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
#################################################
# Flask Routes
#################################################
keys = {
    "1":"Sweet! You’re approved for this loan with an interest rate that is less than 10%!",
    "2":"Congrats! Your loan is approved and your interest rate is in the range between 10% and 14%",
    "3":"Congratulations on being approved for this loan. Your interest rate repayment will be between 14% and 18%",
    "4":"Damn son! You’ve got your loan, but careful, you’re expected to pay back at an interest rate of 18% or more"
}

@app.route("/")
def welcome():
    """List all available api routes."""
    return render_template("index.html")

@app.route("/form",methods=["POST"])
def form():
    annual_inc=request.form["annual_inc"]
    fico_range_low=request.form["fico_range_low"]
    term=request.form["term"]
    loan_amnt=request.form["loan_amnt"]
    #data = [annual_inc, fico_range_low, term, loan_amnt] #request.get_json() #(force=True)
    data=pd.DataFrame({"annual_inc": [annual_inc], "fico_range_low": [fico_range_low], "term": [term], "loan_amnt": [loan_amnt]})
    print(data)
    X=pd.read_csv('features.csv')
    data_df=pd.concat([X,data],ignore_index=True)
    scaler = StandardScaler()
    
    data_scaled = scaler.fit_transform(data_df)
    print([data_scaled[-1]])
    prediction = model.predict([data_scaled[-1]]) 
    gif=""
    output = {"prediction":keys[str(prediction[0])]}
    if(prediction[0] == 1):
        gif="https://media.giphy.com/media/LCdPNT81vlv3y/giphy.gif"
    elif(prediction[0] == 2):
        gif="https://media.giphy.com/media/3orieM0nCVXSnlgKXe/giphy.gif"
    elif(prediction[0] == 3):
        gif="https://media.giphy.com/media/3o6MbuopRH7WatFwC4/giphy.gif"
    else:
        gif="https://media.giphy.com/media/80TEu4wOBdPLG/giphy.gif"    
    return render_template("index.html",output=output, gif=gif)

    

   # return jsonify(output)
   # return jsonify(data)







if __name__ == '__main__':
    app.run(debug=True)
