from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model_nb.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
     if request.method == 'POST':
            
        MonthlyIncome = int(request.form['MonthlyIncome'])
        MonthlyRate = int(request.form['MonthlyRate'])
        DailyRate = int(request.form['DailyRate'])
        YearsAtCompany = float(request.form['YearsAtCompany'])
        YearsInCurrentRole = float(request.form['YearsInCurrentRole'])
        Age = int(request.form['Age'])
        OverTime = int(request.form['OverTime'])
        DistanceFromHome = float(request.form['DistanceFromHome'])
        PercentSalaryHike = float(request.form['PercentSalaryHike'])
        
        
        data = np.array([[MonthlyIncome, MonthlyRate,
                          DailyRate,YearsAtCompany,YearsInCurrentRole,Age,OverTime,DistanceFromHome,PercentSalaryHike]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
            

if  __name__ == '__main__':
    app.run(debug = True)