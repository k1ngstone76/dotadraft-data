import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from joblib import dump


def load_data():
    def load_raw_data():
        data = list()
        with open("data.csv", "r") as f:
            csv_reader = csv.DictReader(f)
            for move in csv_reader:
                data.append(move)

        return data

    X = list()
    y = list()
    for m in load_raw_data():
        move = list()
        for h_id in m["na"].split("-"):  # Banned and taken heroes
            if not h_id:
                continue
            move.append(int(h_id))
        X.append(move)

        y.append(int(m["hero"]))

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

    clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)

    K = 3
    score = top_k_accuracy_score(y_test, y_pred, labels=clf.classes_, k=K)
    print("TOP {} accuracy score: {}".format(K, score))

    dump(clf, 'model.joblib')
