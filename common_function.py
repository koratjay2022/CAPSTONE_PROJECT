import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import is_classifier, is_regressor

class common_functions : 
    
    def __init__(self) :
        pass
    
    def clean_dataframe(self, data):
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        return data

    def replace_values(self, data, column: str, to_replace: dict):
        data[column].replace(to_replace, inplace=True)
        return data

    def remove_rows(self, data, column: str, value):
        data = data[data[column] != value]
        return data

    def label_encode(self, data, column: str):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        return data[column]

    def all_label_encode(self, data):
        le = LabelEncoder()
        object_counlm = data.select_dtypes(include=["object"]).columns
        for i in object_counlm :
             data[i] = le.fit_transform(data[i])
        return data

    def scale_features(self, data, columns: list):
        scaler = StandardScaler()
        data[columns] = scaler.fit_transform(data[columns])
        return data

    def remove_outliers(self, data, column: str, threshold: float = 1.5):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        return data
        
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    if is_classifier(model):
        metric = accuracy_score(y_test, y_pred)
        print(f"Model Type: Classifier\nAccuracy: {metric}")
    elif is_regressor(model):
        metric = r2_score(y_test, y_pred)
        print(f"Model Type: Regressor\nR² Score: {metric}")
    else:
        raise ValueError("Model must be either a classifier or a regressor.")
    
    return y_pred,metric,model

from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import is_classifier, is_regressor

def train_and_evaluate_multiple_models(models, X_train, X_test, y_train, y_test, return_y_pred=False):
    results = {}
    all_predictions = {}  # To store predictions for each model
    
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        model_name = model.__class__.__name__
        correct_predictions = (y_pred == y_test)  # For classification: True where predictions are correct
        
        if is_classifier(model):
            metric = accuracy_score(y_test, y_pred)
            print(f"Model: {model_name} (Classifier) -> Accuracy: {metric}")
        elif is_regressor(model):
            metric = r2_score(y_test, y_pred)
            print(f"Model: {model_name} (Regressor) -> R² Score: {metric}")
        else:
            raise ValueError(f"Model {model_name} is neither a classifier nor a regressor.")
        
        results[model_name] = metric
        all_predictions[model_name] = {
            'y_pred': y_pred,
            'correct_predictions': correct_predictions
        }
    
    if return_y_pred:
        return results, all_predictions
    else:
        return results


# def train_and_evaluate_multiple_models(models, X_train, X_test, y_train, y_test,return_y_pred,) :
#     results = {}
    
#     for model in models:
#         model.fit(X_train, y_train)
        
#         y_pred = model.predict(X_test)
        
#         model_name = model.__class__.__name__
        
#         if is_classifier(model):
#             metric = accuracy_score(y_test, y_pred)
#             print(f"Model: {model_name} (Classifier) -> Accuracy: {metric}")
#         elif is_regressor(model):
#             metric = r2_score(y_test, y_pred)
#             print(f"Model: {model_name} (Regressor) -> R² Score: {metric}")
#         else:
#             raise ValueError(f"Model {model_name} is neither a classifier nor a regressor.")
        
#         results[model_name] = metric
    
#     if return_y_pred:
#         return results,y_pred
#     else :
#         return results