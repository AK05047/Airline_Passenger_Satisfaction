from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from common import RANDOM_STATE, build_preprocessor, load_data, run_and_save

X_train, y_train, X_test, y_test = load_data()

# KNN stores the whole training set, so a 20k stratified subsample keeps the
# saved model small (GitHub / Streamlit Cloud friendly) and prediction fast.
SUBSAMPLE = 20000
if len(X_train) > SUBSAMPLE:
    idx = (y_train.groupby(y_train, group_keys=False)
           .apply(lambda s: s.sample(int(SUBSAMPLE * len(s) / len(y_train)),
                                     random_state=RANDOM_STATE)).index)
    X_train, y_train = X_train.loc[idx], y_train.loc[idx]

model = Pipeline([
    ("prep", build_preprocessor(X_train)),
    ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)),
])

if __name__ == "__main__":
    run_and_save("KNN", "knn", model, X_train, y_train, X_test, y_test)
