from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        input_text = request.form["article"]
        cleaned = clean_text(input_text)
        vector = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vector)[0]
        result = "Real News" if prediction == 1 else "Fake News"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
