from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import numpy as np
from catboost import CatBoostRegressor

app = Flask(__name__)
CORS(app)

def convert_to_array(data):
    return [
        int(data['crop']),                     # Crop
        int(data['season']),                   # Season
        int(data['state']),                    # State
        float(data['area']),                   # Area
        float(data['production']),              # Production
        float(data['annual_rainfall']),        # Annual Rainfall
        float(data['fertilizer']),             # Fertilizer
        float(data['pesticide'])               # Pesticide
    ]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit_data', methods=['POST'])
def getYield():
    data = json.loads(request.data)
    values = list(data.values())
    print(data)
    print(values)
    from_file = CatBoostRegressor()
    from_file.load_model("Yiedl_est_Cat_Boost.cbm")
    pred = from_file.predict(values)
    print(pred)
    data = {
        'text' : "some text"
    }
    return jsonify(pred)

if __name__ == '__main__':
    app.run(debug=True)