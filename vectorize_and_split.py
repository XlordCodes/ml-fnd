from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

with open("data.pkl", "rb") as f:
    data = pickle.load(f)
vectorizer = TfidfVectorizer(max_features=5000)
x = vectorizer.fit_transform(data['clean_text']).toarray()
y = data['label'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Training set size:", x_train.shape)
print("Test set size:", x_test.shape)
print("X shape:", x_train.shape)
print("Y shape:", y_train.shape)
with open("x_train.pkl", "wb") as f:
    pickle.dump(x_train, f)
with open("x_test.pkl", "wb") as f:
    pickle.dump(x_test, f)
with open("y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)
with open("tfidf.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

