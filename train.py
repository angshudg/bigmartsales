import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from catboost import CatBoostRegressor


def main(train_file, model_dir):
    # Load training data
    df = pd.read_csv(train_file)
    df['Item_Units_Sold'] = df['Item_Outlet_Sales'] / df['Item_MRP']

    # Fix Fat Content categories
    df['Item_Fat_Content_Flag'] = 'Regular'
    df.loc[df['Item_Fat_Content'].isin(['LF', 'low fat', 'Low Fat']), 'Item_Fat_Content_Flag'] = 'LowFat'
    df['Item_Fat_Content'] = df['Item_Fat_Content_Flag']
    df = df.drop('Item_Fat_Content_Flag', axis=1)

    # Impute Item_Weight
    item_features = ['Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP']
    item_cat = ['Item_Fat_Content', 'Item_Type']
    item_num = ['Item_Visibility', 'Item_MRP']

    item_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), item_cat),
            ('num', 'passthrough', item_num)
        ]
    )

    known_weight = df[df['Item_Weight'].notnull()]
    missing_weight = df[df['Item_Weight'].isnull()]

    weight_model = Pipeline(steps=[
        ('preprocess', item_preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=7))
    ])

    weight_model.fit(known_weight[item_features], known_weight['Item_Weight'])
    if not missing_weight.empty:
        df.loc[df['Item_Weight'].isnull(), 'Item_Weight'] = weight_model.predict(missing_weight[item_features])

    # Impute Outlet_Size
    outlet_features = ['Outlet_Establishment_Year', 'Outlet_Location_Type', 'Outlet_Type']
    outlet_cat = ['Outlet_Location_Type', 'Outlet_Type']
    outlet_num = ['Outlet_Establishment_Year']

    outlet_preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), outlet_cat),
            ('num', 'passthrough', outlet_num)
        ]
    )

    known_size = df[df['Outlet_Size'].notnull()]
    missing_size = df[df['Outlet_Size'].isnull()]

    size_model = Pipeline(steps=[
        ('preprocess', outlet_preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    size_model.fit(known_size[outlet_features], known_size['Outlet_Size'])
    if not missing_size.empty:
        df.loc[df['Outlet_Size'].isnull(), 'Outlet_Size'] = size_model.predict(missing_size[outlet_features])

    # Train CatBoost regressor
    y = df['Item_Outlet_Sales']
    X = df.drop(columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Units_Sold', 'Item_Outlet_Sales'])
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=7)

    cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.01,
        depth=8,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=7,
        od_type='Iter',
        od_wait=50,
        verbose=0
    )

    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, use_best_model=True)

    # Validation performance
    y_pred = cat_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation R²: {r2:.4f}")

    # Save models
    joblib.dump(weight_model, f"{model_dir}/weight_model.pkl")
    joblib.dump(size_model, f"{model_dir}/size_model.pkl")
    cat_model.save_model(f"{model_dir}/cat_model.cbm", format="cbm")
    joblib.dump(item_features, f"{model_dir}/item_features.pkl")
    joblib.dump(outlet_features, f"{model_dir}/outlet_features.pkl")
    print(f"Models saved in {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="Path to training CSV file")
    parser.add_argument("--model_dir", required=True, help="Directory to save models")
    args = parser.parse_args()

    main(args.train_file, args.model_dir)
