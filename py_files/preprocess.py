import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

def preprocess(data: pd.DataFrame) -> pd.DataFrame:

    '''
    This function preprocess data for training. A pandas dataframe is expected as argument, and it will return the dataframe 
    with applied modifications. 
    In short the function will:
        - Fill in missing values with the median
        - Encode categorical variables using One-Hot encoding from sklearn
        - Save the encoder for inference
    '''
    
    categorical_columns = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
    numerical_columns = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
                         'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 
                         'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'Churn']

    # Handle missing values by filling them with the median of each column
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())

    # Split DataFrame into Categorical and Numerical DataFrames
    categorical_df = data[categorical_columns]
    numerical_df = data[numerical_columns]

    # Encode Categorical Columns
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_categorical = one_hot_encoder.fit_transform(categorical_df)

    # Save the encoder for later use
    encoder_path = "../models/encoder.joblib"
    dump(one_hot_encoder, encoder_path)

    # Concatenate Encoded and Numerical DataFrames
    encoded_categorical_df = pd.DataFrame(encoded_categorical.toarray(), 
                                           columns=one_hot_encoder.get_feature_names_out(categorical_columns))
    preprocessed_df = pd.concat([encoded_categorical_df, numerical_df], axis=1)

    return preprocessed_df
