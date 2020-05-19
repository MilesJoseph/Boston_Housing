import numpy as np
import pandas as pd

 
CAT_DTYPES={"Id": "int64", "MSSubClass": "int64", "MSZoning": "category", "Street": "category",
            "Alley": "category", "LotShape": "category", "LandContour": "category", "Utilities": "category",
            "LotConfig": "category", "LandSlope":"category", "Neighborhood": "category", "Condition1": "category",
            "Condition2": "category", "BldgType": "category", "HouseStyle": "category", "RoofStyle": "category",
            "RoofMatl": "category", "Exterior1st": "category", "Exterior2nd":"category", "MasVnrType": "category", "ExterQual": "category",
            "ExterCond": "category", "Foundation":"category", "BsmtQual": "category", "BsmtCond": "category", "BsmtExposure":"category", "BsmtFinType1":"category",
            "BsmtFinType2":"category","Heating":"category", "HeatingQC": "category", "CentralAir": "category", "Electrical": "category",
            "KitchenQual":"category", "Functional": "category", "FireplaceQu": "category", "GarageType": "category", "GarageFinish":"category",
            "GarageQual": "category", "GarageCond": "category", "PavedDrive": "category", "PoolQC": "category", "Fence": "category",
            "MiscFeature": "category", "SaleType": "category", "SaleCondition": "category"}


data = pd.read_csv("/Users/milesklingenberg/Documents/UWMSBA/590/Data/train_house-1.csv", dtype=CAT_DTYPES)
 
def create_data_categorical(data):
    for col, col_dtype in CAT_DTYPES.items():
        if col_dtype == "category":
            data[col] = data[col].astype('category').cat.codes
​
​
    return data

 
data = create_data_categorical(data)
SFTotals = data.filter(regex="SF")
data['totalsf'] = SFTotals.sum(axis=1)


 
test = pd.read_csv("/Users/milesklingenberg/Documents/UWMSBA/590/Data/test.csv", dtype=CAT_DTYPES)

 
def create_test_categorical(test):
    for col, col_dtype in CAT_DTYPES.items():
        if col_dtype == "category":
            test[col] = test[col].astype('category').cat.codes
    return test

 
test = create_test_categorical(test)
test_1 = test

 
SFTotals = test.filter(regex="SF")
test['totalsf'] = SFTotals.sum(axis=1)

pd.set_option('display.max_columns', 100)

 
NA_col = pd.DataFrame(data.isna().sum(),columns = ['NA_Count'])
NA_col['% of NA'] = (NA_col.NA_Count/len(data))*100
NA_col.sort_values(by = ['% of NA'], ascending = False, na_position = 'first')

## will subset here
#data = data[['OverallQual', 'totalsf', 'ExterQual', 'SalePrice']]
#test = test[['OverallQual', 'totalsf', 'ExterQual']]


X_train = data.copy().drop('SalePrice', axis = 1)
y_train = data[['SalePrice']]

 
X_train = X_train.apply(lambda x: x.fillna(x.mean()), axis=0)
y_train = y_train.apply(lambda x: x.fillna(x.mean()), axis=0)
test = test.apply(lambda x: x.fillna(x.mean()), axis = 0)

 
#X_train.set_index('Id', inplace = True)
#y_train.set_index('Id', inplace = True)
#test.set_index('Id', inplace = True)

##I am going to look at the columns I had originally tried before bridging back out.
 
from sklearn import metrics, tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from xgboost import XGBRegressor

XGB = XGBRegressor(n_jobs=-1)

# Use a grid over parameters of interest
param_grid = {
     'colsample_bytree':[0.3, 0.8, 1.0],
     'n_estimators':[40 ,80, 150, 200,240, 300],
     'max_depth': [1,2, 3,5, 8]
}
​

CV_XGB = GridSearchCV(estimator=XGB, param_grid=param_grid, cv= 10)
 
%time CV_XGB.fit(X_train, y_train)

best_xgb_model = CV_XGB.best_estimator_
print(CV_XGB.best_score_, CV_XGB.best_params_)

pred_train_xgb = best_xgb_model.predict(X_train)
pred_test_xgb = best_xgb_model.predict(test)

 
print(metrics.mean_squared_log_error(y_train, pred_train_xgb).round(5))
print(sqrt(mean_squared_error(y_train, pred_train_xgb)))

 
y_pred_test_xgb = pd.DataFrame(pred_test_xgb, columns = ['SalePrice'])

 
y_pred_test_xgb['Id'] = test_1['Id']

 
columnsTitles = ['Id', 'SalePrice']
​
submission = y_pred_test_xgb.reindex(columns=columnsTitles)
submission .head()


filename = 'House_Pricing.csv'
submission.to_csv(filename, index=False)

 
​
