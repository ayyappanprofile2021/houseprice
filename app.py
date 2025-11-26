from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if(request.method=='GET'):
        return render_template('home.html')    
    else:
        data=CustomData(
            longitude = request.form.get('longitude'),
            latitude = request.form.get('latitude'),
            housing_median_age = request.form.get('housingmedianage'),
            total_rooms = request.form.get('totalrooms'),
            total_bedrooms = request.form.get('totalbedrooms'),
            population = request.form.get('population'),
            households = request.form.get('households'),
            median_income = request.form.get('mediaincome'),
            ocean_proximity = request.form.get('oceanproximity')    
        )

        data_for_prediction = data.get_data_as_data_frame()
        print(data_for_prediction)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(data_for_prediction)

        return render_template('home.html', results=results[0])

if __name__ == '__main__':
    # Run the app and can be accessed using http://localhost:5000
    app.run(host="0.0.0.0", debug=True)