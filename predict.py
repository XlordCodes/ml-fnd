import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

input_text = input("\nPaste your news article text:\n")

cleaned_text = clean_text(input_text)
vectorized = vectorizer.transform([cleaned_text]).toarray()

print("Non-zero TF-IDF features:", (vectorized != 0).sum())

prediction = model.predict(vectorized)[0]

print("\n Prediction Result:")
if prediction == 1:
    print("This article is REAL")
else:
    print("This article is FAKE.")
