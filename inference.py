import argparse
import pandas as pd
import joblib
from catboost import CatBoostRegressor


def main(test_file, model_dir, submission_file):
    # Load test data
    test = pd.read_csv(test_file)

    # Load models and features
    weight_model = joblib.load(f"{model_dir}/weight_model.pkl")
    size_model = joblib.load(f"{model_dir}/size_model.pkl")
    item_features = joblib.load(f"{model_dir}/item_features.pkl")
    outlet_features = joblib.load(f"{model_dir}/outlet_features.pkl")

    cat_model = CatBoostRegressor()
    cat_model.load_model(f"{model_dir}/cat_model.cbm", format="cbm")

    # Handle missing Item_Weight
    missing_weight = test[test['Item_Weight'].isnull()]
    if not missing_weight.empty:
        test.loc[test['Item_Weight'].isnull(), 'Item_Weight'] = weight_model.predict(missing_weight[item_features])

    # Handle missing Outlet_Size
    missing_size = test[test['Outlet_Size'].isnull()]
    if not missing_size.empty:
        test.loc[test['Outlet_Size'].isnull(), 'Outlet_Size'] = size_model.predict(missing_size[outlet_features])

    # Predict
    X_test = test.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
    y_pred = cat_model.predict(X_test)

    # Save submission
    test['Item_Outlet_Sales'] = y_pred
    test[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']].to_csv(submission_file, index=False)
    print(f"Submission saved to {submission_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True, help="Path to test CSV file")
    parser.add_argument("--model_dir", required=True, help="Directory containing trained models")
    parser.add_argument("--submission_file", required=True, help="Path to save submission CSV file")
    args = parser.parse_args()

    main(args.test_file, args.model_dir, args.submission_file)
