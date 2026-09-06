import pandas as pd
import joblib
import typing



def scale_probs(X, model_path = '../models/xgbmodel1.pkl' ):
    model = joblib.load(model_path)
    pred_prob = model.predict_proba
    return 
    
def expected_loss(X):
    pred_prob = X['scaled_probs']
    lgd = X['funded_amnt']
    el = pred_prob * lgd
    return sum(el)

def npv(X):
    return

def survival(X):
    # select columns that matter and values to be displayed 
    # in survival analysis for selected period or cohort
    df = X.copy()
    return df

