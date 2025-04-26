from flask import Flask, request
import numpy as np
import pandas as pd
import pickle
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

# Load your trained random forest weather prediction model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict_weather', methods=["GET"])
def predict_weather():
    """Predict the weather condition ☁️🌧️
    ---
    parameters:
      - name: temperature
        in: query
        type: number
        required: true
      - name: humidity
        in: query
        type: number
        required: true
      - name: pressure
        in: query
        type: number
        required: true
      - name: wind_speed
        in: query
        type: number
        required: true
      - name: visibility
        in: query
        type: number
        required: true
      - name: dew_point
        in: query
        type: number
        required: true
      - name: cloud_cover
        in: query
        type: number
        required: true
      - name: solar_radiation
        in: query
        type: number
        required: true
    responses:
      200:
        description: Weather prediction result
    """
    temperature = float(request.args.get("temperature"))
    humidity = float(request.args.get("humidity"))
    pressure = float(request.args.get("pressure"))
    wind_speed = float(request.args.get("wind_speed"))
    visibility = float(request.args.get("visibility"))
    dew_point = float(request.args.get("dew_point"))
    cloud_cover = float(request.args.get("cloud_cover"))
    solar_radiation = float(request.args.get("solar_radiation"))

    features = np.array([[temperature, humidity, pressure, wind_speed, visibility, dew_point, cloud_cover, solar_radiation]])
    prediction = model.predict(features)[0]

    return f"🌤️ Predicted Weather Value: {prediction}"

@app.route('/predict_weather_file', methods=["POST"])
def predict_weather_file():
    """Batch weather prediction from CSV file
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Batch prediction results
    """
    file = request.files.get("file")
    df = pd.read_csv(file)

    # Make sure the file has exactly 8 columns
    if df.shape[1] != 8:
        return "❌ Error: Uploaded file must have exactly 8 features."

    preds = model.predict(df)
    return str(list(preds))

if __name__ == '__main__':
    app.run(debug=True)