
import pandas as pd
from joblib import load
from py_files.preprocess import preprocess


def make_predictions(input_data: pd.DataFrame) -> dict:
    encoder_path = "../models/encoder.joblib"
    model_path = "../models/model.joblib"
    
    encoder = load(encoder_path)
    model = load(model_path)
    
    preprocessed_data = preprocess(input_data)
    X = preprocessed_data.drop(columns='Churn')
    
    predictions = model.predict(X)

    
    return {'predictions': predictions}
