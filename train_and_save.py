import re
import string
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_and_save():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake["class"] = 0
    true["class"] = 1
    fake = fake.iloc[:-10].copy()
    true = true.iloc[:-10].copy()
    fake = fake[["text", "class"]]
    true = true[["text", "class"]]

    bbc = pd.read_csv("bbc_news.csv")
    bbc_true = pd.DataFrame({
        "text": (bbc["title"] + ". " + bbc["description"]).fillna(""),
        "class": 1
    })

    fake2 = pd.read_csv("fake2.csv")
    fake_extra = pd.DataFrame({
        "text": (fake2["title"] + ". " + fake2["text"]).fillna(""),
        "class": 0
    })

    data = pd.concat([fake, true, bbc_true, fake_extra], ignore_index=True)
    data.drop_duplicates(subset="text", inplace=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data["text"] = data["text"].apply(clean_text)

    x = data["text"]
    y = data["class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)

    pd.DataFrame({"text": x_train, "label": y_train}).to_csv("train.csv", index=False)
    pd.DataFrame({"text": x_test, "label": y_test}).to_csv("test.csv", index=False)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    models = {
        "lr": LogisticRegression(max_iter=1000),
        "dt": DecisionTreeClassifier(random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
        "rf": RandomForestClassifier(random_state=42),
    }

    for name, model in models.items():
        model.fit(xv_train, y_train)
        with open(f"{name}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"\n{name.upper()} Report:\n", classification_report(y_test, model.predict(xv_test)))

if __name__ == "__main__":
    train_and_save()
