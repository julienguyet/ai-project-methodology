import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
    numerical_columns = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
                         'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 
                         'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'Churn']
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
    categorical_df = data[categorical_columns]
    numerical_df = data[numerical_columns]
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_categorical = one_hot_encoder.fit_transform(categorical_df)
    encoded_categorical_df = pd.DataFrame(encoded_categorical.toarray(), 
                                           columns=one_hot_encoder.get_feature_names_out(categorical_columns))
    preprocessed_df = pd.concat([encoded_categorical_df, numerical_df], axis=1)
    return preprocessed_df


def build_model(data: pd.DataFrame) -> dict:
    X = data.drop(columns='Churn')
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)
    log_reg = LogisticRegression(solver='lbfgs', max_iter=10000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    encoder_path = "C:/Users/Mali/ai-project-methodo-draft2/models/encoder.joblib" #The absolute paths are necessary here
    dump(OneHotEncoder, encoder_path)
    model_path = "C:/Users/Mali/ai-project-methodo-draft2/models/model.joblib"
    dump(log_reg, model_path)
    return log_reg, OneHotEncoder, accuracy,report, X_train, X_test, y_train, y_test


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    encoder_path = "C:/Users/Mali/ai-project-methodo-draft2/models/encoder.joblib"
    model_path = "C:/Users/Mali/ai-project-methodo-draft2/models/model.joblib"
    encoder = load(encoder_path)
    model = load(model_path)
    preprocessed_input = preprocess(input_data)
    predictions = model.predict(preprocessed_input.drop(columns='Churn'))
    return predictions


def main():
    df = pd.read_csv('C:/Users/Mali/ai-project-methodo-draft2/data/Dataset/ECommerce.csv')
    data = preprocess(df)
    model, OneHotEncoder, accuracy ,report, X_train, X_test, y_train, y_test = build_model(data)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and model with MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

if __name__ == '__main__':
    main()