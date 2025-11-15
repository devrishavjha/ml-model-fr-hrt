from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# Load models
model_dir = "models"
models = {}
for name in ["LogisticRegression", "DecisionTree", "RandomForest", "SVM"]:
    path = os.path.join(model_dir, f"{name}_model.pkl")
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

# API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [
        data["age"],
        data["sex"],
        data["ChestPainType"],
        data["RestingBP"],
        data["Cholesterol"],
        data["FastingBS"],
        data["RestingECG"],
        data["MaxHR"],
        data["ExerciseAngina"],
        data["Oldpeak"],
        data["ST_Slope"]
    ]
    X = np.array(features).reshape(1, -1)

    results = {}
    for idx, (name, model) in enumerate(models.items(), start=1):
        pred = model.predict(X)[0]
        # Check if model has predict_proba (SVM may not)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]  # probability for class 1
        else:
            # fallback for models like SVM without probability
            prob = float(pred)  # just return predicted class as probability
        results[f"{idx}-{name}"] = {"prediction": int(pred), "probability": round(float(prob), 2)}

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
