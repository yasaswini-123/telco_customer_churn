# Telco Customer Churn Prediction

> **Mini project built during my internship at SURE ProEd** (a unit of SURE Trust, Skill Upgradation for Rural-youth Empowerment).
> Mentor: `Mr. Vamsi Krishna` | Author: `Balla Yasaswini`

A machine learning project that predicts whether a telecom customer is likely to **churn** (leave the company). It trains and compares a **Random Forest**, a **tuned Random Forest** (hyperparameter search with `GridSearchCV`) and a **Decision Tree** on the IBM Telco Customer Churn dataset, and evaluates them with accuracy, precision, recall, F1-score and confusion matrices.

## Objective

Telecom companies lose revenue when customers leave. If the customers who are likely to churn can be identified early, the company can contact them with offers or support. This project builds and compares classification models that flag such customers.

## Workflow

```
 Telco customer CSV (7,043 customers, 21 columns)
        |
        v
 Clean: drop customerID, convert TotalCharges to numbers, encode Churn (No = 0, Yes = 1)
        |
        v
 Stratified 80/20 train-test split
        |
        v
 Preprocessing pipeline
   numeric columns      -> median imputation
   categorical columns  -> most-frequent imputation + one-hot encoding
        |
        v
 Models: Random Forest | Tuned Random Forest (GridSearchCV, F1) | Decision Tree
        |
        v
 Evaluation on the held-out test set
```

Preprocessing and the model are combined into one scikit-learn `Pipeline`, so the same steps are applied to training and test data and no information from the test set leaks into training.

## Project structure

```
telco_customer_churn/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # dataset
├── assignment1_telcom_customer.ipynb      # data preparation, training, tuning, evaluation
├── requirements.txt
└── README.md
```

## Setup and run

Python 3.10 or newer is recommended.

```bash
# 1. clone the repository
git clone <your-repository-url>
cd telco_customer_churn

# 2. create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. install the requirements
pip install -r requirements.txt

# 4. open the notebook
jupyter notebook assignment1_telcom_customer.ipynb
```

In the second code cell of the notebook, change the `FILE` variable so that it points to the CSV file in this folder (for example `FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"`). Then run all cells from top to bottom. Results are reproducible because the split and the models use `random_state=42`, though numbers can differ slightly between scikit-learn versions.

## Dataset

**Telco Customer Churn** (IBM sample dataset, available on Kaggle): https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Licence: `[add the licence shown on the Kaggle page]`

| Item | Value |
|---|---|
| Customers (rows) | 7,043 |
| Columns | 21 (customer ID, 19 features, 1 target) |
| Duplicate rows | 0 |
| Customers who churned | 1,869 (26.5%) |
| Customers who stayed | 5,174 (73.5%) |

The features describe each customer:

- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Account:** tenure (months), Contract, PaperlessBilling, PaymentMethod
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Charges:** MonthlyCharges, TotalCharges
- **Target:** Churn (Yes / No)

### Data preparation

- `customerID` is removed because it only identifies a customer and carries no pattern.
- `TotalCharges` is stored as text in the CSV. It is converted to numbers, and the **11 blank values** that this exposes are filled with the median.
- `Churn` is converted to 0 (stayed) and 1 (churned).
- The 4 numeric features (SeniorCitizen, tenure, MonthlyCharges, TotalCharges) and 15 categorical features are processed separately, as shown in the workflow.
- The data is split 80/20 with stratification, giving **5,634 training** and **1,409 test** customers, with the same churn share in both.
- Because only 26.5% of customers churn, all models use `class_weight="balanced"` so the churn class is not ignored.

## Models

| Model | Settings |
|---|---|
| Random Forest (baseline) | `n_estimators=20`, `class_weight="balanced"`, `random_state=42` |
| Random Forest (tuned) | `GridSearchCV` with 5-fold cross-validation, scored on F1, over `n_estimators` (50, 100, 200), `max_depth` (5, 10, None) and `min_samples_split` (2, 5, 10): 27 combinations, 135 fits |
| Decision Tree | default settings, `class_weight="balanced"`, `random_state=42` |

The best tuned settings were `n_estimators=50`, `max_depth=10` and `min_samples_split=10`, with a cross-validation F1-score of 0.635.

## Results

Evaluated on the 1,409 held-out test customers. Precision, recall and F1-score are for the **churn** class, which is the one the company cares about.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Random Forest (baseline) | 77.4% | 0.60 | 0.45 | 0.51 |
| Random Forest (tuned) | 76.3% | 0.54 | 0.73 | 0.62 |
| Decision Tree | 72.8% | 0.49 | 0.48 | 0.49 |

Confusion matrices (rows are the actual class, columns the prediction; stayed first, churned second):

| Model | Stayed predicted correctly | Stayed flagged as churn | Churn missed | Churn caught |
|---|---|---|---|---|
| Random Forest (baseline) | 922 | 113 | 206 | 168 |
| Random Forest (tuned) | 801 | 234 | 100 | 274 |
| Decision Tree | 845 | 190 | 193 | 181 |

### What the results show

- **Accuracy alone is misleading here.** A model that predicted "stays" for every customer would already score 73.5%, so all three models improve only modestly on that on this measure.
- **The tuned Random Forest is the most useful model for finding churners.** Its accuracy is about one point lower than the baseline, but it catches **274 of the 374 churners (73%)**, compared with 168 (45%) for the baseline. Tuning was scored on F1, which rewards finding churners.
- **The cost is more false alarms.** The tuned model flags 234 loyal customers as likely churners, against 113 for the baseline. Whether that is acceptable depends on what a retention offer costs compared with losing a customer.
- **The Decision Tree is the weakest model.** A single unrestricted tree memorises the training data (99.8% accuracy on the training set, but only 72.8% on the test set) and generalises worse than the forest.

## Limitations and future work

- **One train-test split.** Results come from a single 80/20 split of 1,409 test customers, so small differences between models (for example 77.4% and 76.3%) may be noise. Reporting cross-validated scores would be more reliable.
- **No exploratory analysis or feature importance yet.** Adding charts (churn by contract type, tenure and monthly charges) and a feature-importance plot would show which factors drive churn.
- **No saved model or app.** The trained pipeline could be saved with `joblib` and used in a small Streamlit app where a user enters customer details and sees the churn risk.
- **Other models to compare:** Logistic Regression, Gradient Boosting and XGBoost, which often perform well on this kind of data.
- **Decision threshold and imbalance handling.** Tuning the probability threshold, or trying SMOTE, could improve the balance between caught churners and false alarms.

## Libraries used

pandas, scikit-learn, matplotlib, seaborn, Jupyter Notebook.

## Acknowledgements

- **SURE ProEd** (a unit of SURE Trust) and my mentor `Mr. Vamsi Krishna` for guidance during the internship.
- IBM, for the Telco Customer Churn sample dataset, and the Kaggle uploader.
- Libraries: pandas, scikit-learn, matplotlib, seaborn.

## Author

`Balla Yasaswini` | [GitHub](https://github.com/yasaswini-123) | [LinkedIn](https://www.linkedin.com/in/ballayasaswini/)

## Licence

To be confirmed with the mentor. The dataset licence is listed above and applies to the data, not to this code.
