from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import json
import re

import google.generativeai as genai

app = Flask(__name__, template_folder='templates')
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def extract_json(text):
    json_text = re.search(r'\{.*\}', text, re.DOTALL)
    if json_text:
        return json.loads(json_text.group())
    else:
        raise ValueError("No valid JSON found")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    prompt = f"""
    Analyze this soil data:
    - Nitrogen: {data['N']}
    - Phosphorus: {data['P']}
    - Potassium: {data['K']}
    - pH: {data['ph']}
    - Temperature: {data['temperature']} °C
    - Humidity: {data['humidity']} %
    - Rainfall: {data['rainfall']} mm
    
    Based on this, recommend:
    1. The best crop
    2. Soil health score (0-100)
    3. One soil improvement tip
    4. Growth difficulty (Easy, Medium, Hard)
    5. Best season
    6. 2-3 companion crops
    7. Create a customized natural fertilizer recipe using simple farm ingredients (like cow dung, banana peel, neem oil, compost, bone meal, etc.) that suits the soil condition. Mention quantities clearly.


    Respond only in pure JSON format:
    {{
        "crop_recommendation": "...",
        "soil_health_score": 90,
        "improvement_tip": "...",
        "growth_difficulty": "...",
        "best_season": "...",
        "companion_crops": ["...", "..."],
        "fertilizer_recipe": "Mix 2 kg cow dung, 500g banana peel powder, and 1 litre neem oil."

    }}
    """

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)

    result = extract_json(response.text)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
