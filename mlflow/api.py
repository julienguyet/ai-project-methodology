from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from sklearn.preprocessing import OneHotEncoder

# Load the preprocessor and model
encoder_path = "C:/Users/Mali/ai-project-methodology/models/encoder.joblib"
model_path = "C:/Users/Mali/ai-project-methodology/models/model.joblib"
encoder = load(encoder_path)
model = load(model_path)

api = Flask(__name__)

@api.route("/predict", methods=['GET']) 
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Invalid file format. Please upload a CSV file"}), 400
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400
    preprocessed_df = preprocess(df)
    predictions = model.predict(preprocessed_df)
    response_data = {
        "predictions": predictions.tolist()
    }
    return jsonify(response_data), 200

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
    numerical_columns = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 
                         'SatisfactionScore', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 
                         'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
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

if __name__ == "__main__":
    api.run(debug=True)

