
from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load trained AI model and TF-IDF
saved = joblib.load("fake_job_detector.pkl")

model = saved["model"]
tfidf = saved["tfidf"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    job = request.form["job_description"]

    cleaned = clean_text(job)

    features = tfidf.transform([cleaned])

    result = model.predict(features)[0]

    if result == 1:
        prediction = "Potentially Fake Job ⚠️"
    else:
        prediction = "Genuine Job Prediction ✅"

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
