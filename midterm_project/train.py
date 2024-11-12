import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder, TargetEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# parameter grid
def get_xgb_parameters():
    return {
        'n_estimators': [100, 200],
        'max_depth': [6, 8, 10],
        'colsample_bytree': [0.7, 0.8],
        'learning_rate': [0.1, 0.01],
        'subsample': [0.8, 1.0],
        'scale_pos_weight': [3.0, 3.04]
    }

# number of folds for GridSearchCV
cv = 5
random_state = 42

# reading and cleaning the data
def load_and_clean_data(filepath):
    df_raw = pd.read_csv(filepath)
    df = df_raw.copy()

    # data cleaning
    df.columns = df.columns.str.lower().str.replace('-', '_')
    categorical_columns = df.select_dtypes(include=['object']).columns
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '').str.replace('-', '_').replace('?', np.nan)

    df.dropna(inplace=True)
    df.drop_duplicates(keep='first', inplace=True)
    df.salary = (df.salary == '>50k').astype('int')
    return df

# data splitting
def split_data(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed)
    
    X_train = df_train.drop(columns='salary').reset_index(drop=True)
    X_val = df_val.drop(columns='salary').reset_index(drop=True)
    X_test = df_test.drop(columns='salary').reset_index(drop=True)
    
    y_train = df_train.salary.values
    y_val = df_val.salary.values
    y_test = df_test.salary.values
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# preprocessing
def get_preprocessor():
    categorical_cols = ['relationship', 'marital_status', 'education', 'occupation']
    numerical_cols = ['hours_per_week', 'capital_loss', 'capital_gain', 'age', 'education_num']

    education_order = [
        'preschool', '1st_4th', '5th_6th', '7th_8th', '9th', '10th', '11th', '12th',
        'hs_grad', 'some_college', 'assoc_acdm', 'assoc_voc', 'bachelors', 'masters', 
        'prof_school', 'doctorate'
    ]
    
    categorical_transformer = ColumnTransformer(transformers=[
        ('relationship_ohe', OneHotEncoder(handle_unknown='ignore'), ['relationship']),
        ('education_ord', OrdinalEncoder(categories=[education_order]), ['education']),
        ('marital_status_te', TargetEncoder(), ['marital_status']),
        ('occupation_te', TargetEncoder(), ['occupation'])
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    return preprocessor

# model training with GridSearchCV
def train_model_with_gridsearch(X_train, y_train, preprocessor):
    xgb_model = xgb.XGBClassifier()
    
    # GridSearchCV to tune hyperparameters
    param_grid = get_xgb_parameters()
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=seed
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', grid_search)
    ])
    
    # fit the model
    pipeline.fit(X_train, y_train)
    
    # best model and best score
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    return pipeline, best_model, best_score

# save the model and preprocessor
def save_model_and_preprocessor(model, preprocessor, filename):
    with open(filename, 'wb') as f:
        pickle.dump((model, preprocessor), f)

# main function to orchestrate the entire workflow
def main():
    # load and split the dataset
    df = load_and_clean_data('./data/salary.csv')    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    preprocessor = get_preprocessor()
    print(f"Training the model with GridSearchCV...")
    
    pipeline, best_model, best_score = train_model_with_gridsearch(X_train, y_train, preprocessor)
    print(f"Best AUC score from GridSearchCV: {best_score:.4f}")
    print('Training the final model...')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC on the test set: {auc:.4f}")
    
    # save the model and preprocessor
    save_model_and_preprocessor(best_model, preprocessor, 'xgboost_model_with_preprocessor.pkl')
    print("Model training and saving completed.")

if __name__ == '__main__':
    main()