import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import pandas as pd


pd.options.display.max_columns = 90


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



def create_dt(is_train=True):
    train = pd.read_csv("/Users/milesklingenberg/Documents/UWMSBA/590/Data/train_house-1.csv", dtype=CAT_DTYPES)
    train = pd.DataFrame(train)
    for col, col_dtype in CAT_DTYPES.items():
        if col_dtype == "category":
            train[col] = train[col].astype('category').cat.codes

    drop_columns = ['']


    dt = train

    return dt

train_df = create_dt(is_train=True)

###i cannnot get a lambda to work for filling all nan, I believe it should only be an issue for float...

train_df['LotFrontage'] = train_df['LotFrontage'].fillna(0)
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(0)

clf = RandomForestClassifier(n_estimators = 1000, random_state=0, n_jobs=-1)

x = train_df.iloc[:, 0:80]
y = train_df["SalePrice"]

clf.fit(x,y)

for feature in zip(list(train_df), clf.feature_importances_):
   print(feature)

### Then we will want to select the best features, I am not going to read that whole list.

feat_labels = list(train_df.iloc[:,0:80])
clf_features = pd.DataFrame(clf.feature_importances_)
feat_labels = pd.DataFrame(feat_labels)

merged_df = [feat_labels, clf_features]

result = pd.concat(merged_df, axis = 1)
result.columns = ['feat', 'importance']

result_1 = result.sort_values(by = 'importance')
print(result_1)
