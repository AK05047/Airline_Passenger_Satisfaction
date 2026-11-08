# Airline Passenger Satisfaction - Classification Models & Streamlit App

## a. Problem statement
Predict whether an airline passenger will be **satisfied** or **neutral/dissatisfied** with their flight, based on passenger details, trip details and ratings of airline services. This is a binary classification problem. Five classification models are trained on the same dataset, compared using six evaluation metrics, and demonstrated in an interactive Streamlit web app.

## b. Dataset description
- **Dataset:** Airline Passenger Satisfaction (Kaggle), provided as `train.csv` and `test.csv`.
- **Size:** 103,904 training rows and 25,976 test rows (about 130k in total).
- **Features:** 22 input features after dropping the index column (`Unnamed: 0`) and the `id` column.
  - *Passenger / trip (6):* Gender, Customer Type, Age, Type of Travel, Class, Flight Distance
  - *Service ratings (14, scale 0-5, 0 = not applicable):* Inflight wifi service, Departure/Arrival time convenient, Ease of Online booking, Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service, Baggage handling, Checkin service, Inflight service, Cleanliness
  - *Delays (2):* Departure Delay in Minutes, Arrival Delay in Minutes (310 missing values in the training set, imputed with the median)
- **Target:** `satisfaction` - `satisfied` (1) vs `neutral or dissatisfied` (0). Classes are fairly balanced: 43.3% satisfied in the training set and 43.9% in the test set.
- **Preprocessing (inside each saved pipeline):** median imputation and standard scaling for numeric columns, one-hot encoding for categorical columns.
- **Note:** KNN and SVM were trained on a stratified 20,000-row subsample of the training data to keep training time and saved-model size manageable. The other three models use the full training set. All models are evaluated on the full test set.

## c. GitHub repository link
`<PASTE YOUR GITHUB REPO LINK HERE>`

Live Streamlit app: `<PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>`

## d. Models used

### Comparison table (evaluated on the full test set, 25,976 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8716 | 0.9256 | 0.8691 | 0.8330 | 0.8507 | 0.7387 |
| Decision Tree | 0.9502 | 0.9881 | 0.9571 | 0.9282 | 0.9424 | 0.8989 |
| KNN | 0.9161 | 0.9674 | 0.9340 | 0.8704 | 0.9011 | 0.8300 |
| Naive Bayes | 0.8447 | 0.9121 | 0.8367 | 0.8028 | 0.8194 | 0.6837 |
| Support Vector Machine | 0.9443 | 0.9837 | 0.9442 | 0.9280 | 0.9360 | 0.8869 |

Precision, Recall and F1 are computed for the positive class (`satisfied`).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A solid, fast and interpretable baseline (accuracy 0.8716, AUC 0.9256). As a linear model it cannot capture interactions between features, and its recall (0.8330) shows it misses about 17% of satisfied passengers. It clearly trails the non-linear models. |
| Decision Tree | The best model on every metric (accuracy 0.9502, AUC 0.9881, MCC 0.8989). It captures non-linear rules and feature interactions that the linear and probabilistic models miss. Depth was limited (`max_depth=12`, `min_samples_leaf=10`) to reduce overfitting. Its most important features were Online boarding, Inflight wifi service and Type of Travel. |
| KNN | Good performance (accuracy 0.9161, AUC 0.9674) but below the tree and SVM, with the lowest recall of the top three (0.8704). Trained on a 20k subsample, so it may improve with more data. It also needs the full training set at prediction time and is sensitive to scaling and the choice of k. |
| Naive Bayes | The weakest model (accuracy 0.8447, MCC 0.6837). Its assumption that features are independent does not hold here, because the service ratings are strongly correlated, and one-hot encoded and ordinal rating features are not Gaussian. It is very fast, but accuracy is the trade-off. |
| Support Vector Machine | A close second (accuracy 0.9443, AUC 0.9837, MCC 0.8869), within about 0.6 percentage points of the Decision Tree while trained on only 20k rows. The RBF kernel captures non-linear boundaries well, but training and prediction are slower than for the other models. |
| **Overall, Winner for your dataset?** | **Decision Tree** - highest Accuracy, AUC, Precision, Recall, F1 and MCC. The SVM is a very close runner-up, followed by KNN, Logistic Regression and Naive Bayes. |

## Streamlit app features
- Upload a test CSV (it must contain all input features and the `satisfaction` column; `Unnamed: 0` and `id` are ignored if present).
- Dropdown to select one of the five models.
- Display of Accuracy, AUC, Precision, Recall, F1 and MCC.
- Confusion matrix, classification report and ROC curve.
- "Compare all models" tab showing all five models side by side on the uploaded data.
- Models are loaded from `model/*.joblib`; nothing is retrained in the app.

## Project structure
```
project-folder/
|-- app.py                  # Streamlit app
|-- requirements.txt
|-- README.md
|-- test_data.csv           # test data to upload in the app
|-- data/train.csv          # training data (used only by the training scripts)
|-- model/
|   |-- common.py           # shared preprocessing + evaluation helpers
|   |-- logistic_regression.py, decision_tree.py, knn.py, naive_bayes.py, svm.py
|   |-- *.joblib            # saved trained models (preprocessing included)
|   |-- *_metrics.json      # metrics for each model on test_data.csv
```

## Run locally
```bash
pip install -r requirements.txt
# optional: retrain (each script saves its model into model/)
cd model && python logistic_regression.py && python decision_tree.py && python knn.py && python naive_bayes.py && python svm.py && cd ..
streamlit run app.py
```
