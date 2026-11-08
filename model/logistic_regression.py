from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import RANDOM_STATE, build_preprocessor, load_data, run_and_save

X_train, y_train, X_test, y_test = load_data()

model = Pipeline([
    ("prep", build_preprocessor(X_train)),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])

if __name__ == "__main__":
    run_and_save("Logistic Regression", "logistic_regression",
                 model, X_train, y_train, X_test, y_test)
