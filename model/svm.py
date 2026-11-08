from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from common import RANDOM_STATE, build_preprocessor, load_data, run_and_save

X_train, y_train, X_test, y_test = load_data()

# RBF-kernel SVC scales ~O(n^2), so train on a 20k stratified subsample.
SUBSAMPLE = 20000
if len(X_train) > SUBSAMPLE:
    idx = (y_train.groupby(y_train, group_keys=False)
           .apply(lambda s: s.sample(int(SUBSAMPLE * len(s) / len(y_train)),
                                     random_state=RANDOM_STATE)).index)
    X_train, y_train = X_train.loc[idx], y_train.loc[idx]

# AUC uses decision_function scores, so probability=True (slow) isn't needed.
model = Pipeline([
    ("prep", build_preprocessor(X_train)),
    ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)),
])

if __name__ == "__main__":
    run_and_save("SVM", "svm", model, X_train, y_train, X_test, y_test)
