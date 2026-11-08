from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from common import build_preprocessor, load_data, run_and_save

X_train, y_train, X_test, y_test = load_data()

# GaussianNB: suitable for the scaled numeric + one-hot features here
# (MultinomialNB would need non-negative inputs, which scaling breaks).
model = Pipeline([
    ("prep", build_preprocessor(X_train)),
    ("clf", GaussianNB()),
])

if __name__ == "__main__":
    run_and_save("Naive Bayes", "naive_bayes",
                 model, X_train, y_train, X_test, y_test)
