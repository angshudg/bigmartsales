import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score

from catboost import CatBoostRegressor, Pool


df=pd.read_csv('train_v9rqX0R.csv')
df_orig=df.copy()
df['Item_Units_Sold'] = df['Item_Outlet_Sales'] / df['Item_MRP']

df['Item_Fat_Content_Flag']='Regular'
df.loc[df['Item_Fat_Content'].isin(['LF', 'low fat','Low Fat']),'Item_Fat_Content_Flag']='LowFat'
df['Item_Fat_Content'] = df['Item_Fat_Content_Flag']
df = df.drop('Item_Fat_Content_Flag', axis=1)

## Item_Weight Imputation
item_features = ['Item_Fat_Content',
                 'Item_Visibility',
                 'Item_Type',
                 'Item_MRP']

true_vals = df.loc[df['Item_Weight'].notnull(), 'Item_Weight']
features = df.loc[df['Item_Weight'].notnull(), item_features]

item_cat = ['Item_Fat_Content', 'Item_Type']
item_num = ['Item_Visibility', 'Item_MRP']

mask_idx = np.random.choice(true_vals.index, size=int(0.2*len(true_vals)), replace=False)
masked = features.copy()
masked.loc[mask_idx, :] = masked.loc[mask_idx, :]
fake_missing = true_vals.copy()
fake_missing.loc[mask_idx] = np.nan

item_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), item_cat),
        ('num', 'passthrough', item_num)
    ])

known_weight = df[df['Item_Weight'].notnull()]
missing_weight = df[df['Item_Weight'].isnull()]

weight_model = Pipeline(steps=[
  ('preprocess', item_preprocessor),
  ('regressor', RandomForestRegressor(n_estimators=200, random_state=7))
])

weight_model.fit(known_weight[item_features], known_weight['Item_Weight'])
df.loc[df['Item_Weight'].isnull(), 'Item_Weight'] = weight_model.predict(missing_weight[item_features])

## Outlet_Size Imputation (Classification)
outlet_features = ['Outlet_Establishment_Year',
                   'Outlet_Location_Type',
                   'Outlet_Type']

true_size = df.loc[df['Outlet_Size'].notnull(), 'Outlet_Size']
features_size = df.loc[df['Outlet_Size'].notnull(), outlet_features]

outlet_cat = ['Outlet_Location_Type', 'Outlet_Type']
outlet_num = ['Outlet_Establishment_Year']

outlet_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), outlet_cat),
        ('num', 'passthrough', outlet_num)
    ])

known_size = df[df['Outlet_Size'].notnull()]
missing_size = df[df['Outlet_Size'].isnull()]

size_model = Pipeline(steps=[
    ('preprocess', outlet_preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

size_model.fit(known_size[outlet_features], known_size['Outlet_Size'])
df.loc[df['Outlet_Size'].isnull(), 'Outlet_Size'] = size_model.predict(missing_size[outlet_features])

## Sales Prediction Model Training
y = df['Item_Outlet_Sales']
X = df.drop(columns=['Item_Identifier',
                     'Outlet_Identifier',
                     'Item_Units_Sold',
                     'Item_Outlet_Sales'])

cat_features = X.select_dtypes(include=['object']).columns.tolist()
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=7
)

cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.01,
    depth=8,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=7,
    od_type='Iter',
    od_wait=50
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_features,
    use_best_model=True
)

y_pred = cat_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation R²: {r2:.4f}")

test = pd.read_csv("test_AbJTz2l.csv")

# Missing Value handling
missing_weight = test[test['Item_Weight'].isnull()]
if not missing_weight.empty:
    test.loc[test['Item_Weight'].isnull(), 'Item_Weight'] = weight_model.predict(missing_weight[item_features])

missing_size = test[test['Outlet_Size'].isnull()]
if not missing_size.empty:
    test.loc[test['Outlet_Size'].isnull(), 'Outlet_Size'] = size_model.predict(missing_size[outlet_features])

X_test = test.drop(columns=['Item_Identifier',
                  'Outlet_Identifier'])
cat_features = X_test.select_dtypes(include=['object']).columns.tolist()

y_pred_units = cat_model.predict(X_test)
# test['Item_Units_Sold'] = y_pred_units
# test['Item_Outlet_Sales'] = test['Item_Units_Sold'] * test['Item_MRP']
# test[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']].to_csv("submission_006.csv", index=False)

test['Item_Outlet_Sales'] = y_pred_units
# test['Item_Outlet_Sales'] = test['Item_Units_Sold'] * test['Item_MRP']
test[['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']].to_csv("submission_final.csv", index=False)