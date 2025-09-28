import re
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
real_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")
fake_df['label'] = 0
real_df['label'] = 1
data = pd.concat([real_df, fake_df], ignore_index=True)
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)
data['clean_text'] = data['text'].apply(clean_text)
print(data['text'].iloc[0])
print(data['clean_text'].iloc[0])
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)
