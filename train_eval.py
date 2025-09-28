from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
with open("x_test.pkl", "rb") as f:
    x_test = pickle.load(f)
with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)
with open("x_train.pkl", "rb") as f:
    x_train = pickle.load(f)
with open("y_train.pkl", "rb") as f:
    y_train = pickle.load(f)
def evaluate(model, name):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n {name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

lr=LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
evaluate(lr, "logistic_regression")

nb=MultinomialNB()
nb.fit(x_train, y_train)
evaluate(nb, "MultinomialNB")

sdg=SGDClassifier(loss='hinge')
sdg.fit(x_train, y_train)
evaluate(sdg, "SGD")

with open("best_model.pkl", "wb") as f:
    pickle.dump(lr, f)
