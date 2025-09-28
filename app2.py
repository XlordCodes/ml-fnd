import pickle
from flask import Flask, request, render_template
from train_and_save import clean_text

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
models = {
    "LR": pickle.load(open("lr_model.pkl", "rb")),
    "DT": pickle.load(open("dt_model.pkl", "rb")),
    "GB": pickle.load(open("gb_model.pkl", "rb")),
    "RF": pickle.load(open("rf_model.pkl", "rb"))
}

def predict_news(news_text):
    cleaned = clean_text(news_text)
    vectorized = vectorizer.transform([cleaned])
    return {
        name: "Fake News" if model.predict(vectorized)[0] == 0 else "Real News"
        for name, model in models.items()
    }

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_input = request.form.get("news")
        prediction = predict_news(user_input)
    return render_template("index2.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
