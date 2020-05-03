import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz


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
    """
    This function just reads in the data and cleans the data a little bit and returns a dataframe based
    on the read data.
    """
    train = pd.read_csv("/Users/milesklingenberg/Documents/UWMSBA/590/Data/train_house-1.csv", dtype=CAT_DTYPES)
    train = pd.DataFrame(train)
    for col, col_dtype in CAT_DTYPES.items():
        if col_dtype == "category":
            train[col] = train[col].astype('category').cat.codes

    #drop_columns = ['']
    #was going to drop columns

    dt = train

    return dt

train_df = create_dt(is_train=True)

###i cannnot get a lambda to work for filling all nan, I believe it should only be an issue for float...

train_df['LotFrontage'] = train_df['LotFrontage'].fillna(0)
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(0)


### It turned out it was just an issue for float, thankfully, something to figure out in the future.

##Just to mention here, I am using a random forest classifier to figure out which variables to select. If I were doing this
##in the future I would probably do a linear recursive feauture engineering, but since we are doing decision trees I will
## stick to a tree based approach.


##Setting up the model here for recursive.


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

#This merged data frame has the list of variables ranked by their gini impurity.
## I will start by looking at LotArea, GrLivArea, 1stfloor, GarageArea as these have the best gini purity.
## I will also explore some other options to see if this was incorrect

result = pd.concat(merged_df, axis = 1)
result.columns = ['feat', 'importance']

result_1 = result.sort_values(by = 'importance')
print(result_1)

dec_t = DecisionTreeClassifier(min_samples_split = 20, random_state=99)
dec_t.fit(x,y)

test = pd.read_csv("/Users/milesklingenberg/Documents/UWMSBA/590/Data/train_house-1.csv", dtype=CAT_DTYPES)


def create_test(data):
    """"
    bringing in and cleaning the test data.
    """
    test = pd.read_csv("/Users/milesklingenberg/Documents/UWMSBA/590/Data/train_house-1.csv", dtype=CAT_DTYPES)
    for col, col_dtype in CAT_DTYPES.items():
        if col_dtype == "category":
            test[col]= test[col].astype("category").cat.codes


    return test

test_1 = create_test(test)

test_1 = test[['LotArea', 'GrLivArea', '1stFlrSF', 'GarageArea']]
test_1 = pd.DataFrame(test_1)
print(test_1.head(10))
