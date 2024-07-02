# Importing the required packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import mlflow
import os
from sklearn.impute import SimpleImputer

RANDOM_SEED =  42
# Load the dataset
file_path = 'train.csv'  # Replace with the correct file path
data = pd.read_csv(file_path)

# Drop 'Loan_ID' column as it's not needed for the prediction
data = data.drop(columns=['Loan_ID'])

# Split the data into training and testing sets
X = data.drop(columns=['Loan_Status'])
y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_train[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']] = imputer.fit_transform(X_train[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']] = imputer.transform(X_test[['LoanAmount', 'Loan_Amount_Term', 'Credit_History']])

# Handle outliers using IQR method
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df

continuous_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

for column in continuous_columns:
    X_train = handle_outliers(X_train, column)
    X_test = handle_outliers(X_test, column)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

for column in categorical_columns:
    X_train[column] = label_encoder.fit_transform(X_train[column])
    X_test[column] = label_encoder.transform(X_test[column])

# Encode the target variable
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# RandomForest
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators': [200,400, 700],
    'max_depth': [10,20,30],
    'criterion' : ["gini", "entropy"],
    'max_leaf_nodes': [50, 100]
}

grid_forest = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_forest, 
        cv=5, 
        n_jobs=-1, 
        scoring='accuracy',
        verbose=0
    )
model_forest = grid_forest.fit(X_train, y_train)

#Logistic Regression

lr = LogisticRegression(random_state=RANDOM_SEED)
param_grid_log = {
    'C': [100, 10, 1.0, 0.1, 0.01],
    'penalty': ['l1','l2'],
    'solver':['liblinear']
}

grid_log = GridSearchCV(
        estimator=lr,
        param_grid=param_grid_log, 
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=0
    )
model_log = grid_log.fit(X_train, y_train)

#Decision Tree

dt = DecisionTreeClassifier(
    random_state=RANDOM_SEED
)

param_grid_tree = {
    "max_depth": [3, 5, 7, 9, 11, 13],
    'criterion' : ["gini", "entropy"],
}

grid_tree = GridSearchCV(
        estimator=dt,
        param_grid=param_grid_tree, 
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=0
    )
model_tree = grid_tree.fit(X_train, y_train)

mlflow.set_experiment("Loan_prediction")

# Model evelaution metrics
def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # Close plot
    plt.close()
    return(accuracy, f1, auc)


def mlflow_logging(model, X, y, name):
    
     with mlflow.start_run() as run:
        #mlflow.set_tracking_uri("http://0.0.0.0:5001/")
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)      
        pred = model.predict(X)
        #metrics
        (accuracy, f1, auc) = eval_metrics(y, pred)
        # Logging best parameters from gridsearch
        mlflow.log_params(model.best_params_)
        #log the metrics
        mlflow.log_metric("Mean CV score", model.best_score_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("AUC", auc)

        # Logging artifacts and model
        mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model, name)
        
        mlflow.end_run()

mlflow_logging(model_tree, X_test, y_test, "DecisionTreeClassifier")
mlflow_logging(model_log, X_test, y_test, "LogisticRegression")
mlflow_logging(model_forest, X_test, y_test, "RandomForestClassifier")
