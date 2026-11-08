from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import RANDOM_STATE, build_preprocessor, load_data, run_and_save

X_train, y_train, X_test, y_test = load_data()

# max_depth / min_samples_leaf limit overfitting
model = Pipeline([
    ("prep", build_preprocessor(X_train)),
    ("clf", DecisionTreeClassifier(max_depth=12, min_samples_leaf=10,
                                   random_state=RANDOM_STATE)),
])

if __name__ == "__main__":
    run_and_save("Decision Tree", "decision_tree",
                 model, X_train, y_train, X_test, y_test)
