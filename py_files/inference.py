
import pandas as pd
from joblib import load
from py_files.preprocess import preprocess


def make_predictions(input_data: pd.DataFrame) -> dict:

    '''
    This is our final function to run predictions on unseen data. It is expecting a pandas dataframe as input.
    The function loads the encoder and model from joblib files defined using previous functions.
    The returned result is a dictionnary of the predictions. 
    '''

    encoder_path = "../models/encoder.joblib"
    model_path = "../models/model.joblib"
    
    encoder = load(encoder_path)
    model = load(model_path)
    
    preprocessed_data = preprocess(input_data)
    X = preprocessed_data.drop(columns='Churn')
    
    predictions = model.predict(X)

    
    return {'predictions': predictions}
