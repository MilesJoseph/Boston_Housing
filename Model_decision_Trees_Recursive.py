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

# ##i cannnot get a lambda to work for filling all nan, I believe it should only be an issue for float...

train_df['LotFrontage'] = train_df['LotFrontage'].fillna(0)
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(0)


# ## It turned out it was just an issue for float, thankfully, something to figure out in the future.

# Just to mention here, I am using a random forest classifier to figure out which variables to select. If I were doing this
# in the future I would probably do a linear recursive feauture engineering, but since we are doing decision trees I will
# stick to a tree based approach.


# Setting up the model here for recursive.


clf = RandomForestClassifier(n_estimators = 1000, random_state=0, n_jobs=-1)

x = train_df.iloc[:, 0:80]
y = train_df["SalePrice"]

clf.fit(x,y)

#for feature in zip(list(train_df), clf.feature_importances_):
   #print(feature)

# Then we will want to select the best features, I am not going to read that whole list.

feat_labels = list(train_df.iloc[:,0:80])
clf_features = pd.DataFrame(clf.feature_importances_)
feat_labels = pd.DataFrame(feat_labels)

merged_df = [feat_labels, clf_features]


# This merged data frame has the list of variables ranked by their gini impurity.
# I will start by looking at LotArea, GrLivArea, 1stfloor, GarageArea as these have the best gini purity.
# I will also explore some other options to see if this was incorrect

result = pd.concat(merged_df, axis = 1)
result.columns = ['feat', 'importance']

result_1 = result.sort_values(by = 'importance')
print(result_1)

dec_t = DecisionTreeClassifier(min_samples_split = 20, random_state=99)
dec_t.fit(x,y)


test = pd.read_csv("/Users/milesklingenberg/Documents/UWMSBA/590/Data/test_house-1.csv", dtype = CAT_DTYPES)

##Okay....this took me a long time to realize... there are spaces in the
## column names for the test set... which is why my dtypes are not working
test['Lot Area'] = test['Lot Area'].fillna(0)
test['Gr Liv Area'] = test['Gr Liv Area'].fillna(0)
test['1st Flr SF'] = test['1st Flr SF'].fillna(0)
test['Garage Area'] = test['Garage Area'].fillna(0)


test_1 = test[['Lot Area', 'Gr Liv Area', '1st Flr SF', 'Garage Area']]
train_df_x = train_df[['LotArea', 'GrLivArea', '1stFlrSF', 'GarageArea']]
train_df_y = train_df['SalePrice']

dec_clipped = DecisionTreeClassifier(min_samples_split = 2, random_state = 99)
dec_clipped_fit = dec_clipped.fit(train_df_x, train_df_y)
dec_clipped_predict = dec_clipped.predict(test_1)
predictions_1_clipped = pd.DataFrame(dec_clipped_predict, columns = ['output'])

# sklearn has a way to export trees graphically

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# dot_data = StringIO()

#export_graphviz(dec_clipped_fit, out_file=dot_data,
                #filled=True, rounded=True,
                #special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())
#graph.write_png("clipped_dec")

# Taking graph out as it takes some time to run.
# You can see that in hte min number of sample split at each one of my nodes I am allowing a split of 2. Therefore the graph will be quite
# large and not really helpful for display.

values = pd.read_excel("/Users/milesklingenberg/Documents/UWMSBA/590/Data/Values.xlsx")

from sklearn.metrics import mean_squared_error
from math import sqrt
print(sqrt(mean_squared_error(values['SalePrice'], predictions_1_clipped['output'])))

# So we get an RMSE of 38,407 for the model that just uses the Random Forest Classifier.
# THis is not good, but I guesss it depends on min and max....also we are using a decision tree..

print(min(train_df['SalePrice']))
print(max(train_df['SalePrice']))

# Right, so on a scale of 34900 and max 755000 for data, 38,407 is not really acceptable.
# Since the sample split in the tree is 2, this means it is really only spliting on itself with
# an average of the split within the last node.

# It would probably be helpful at this point to look at the data as opposed to random forest classifier.

import matplotlib.pyplot as plt
### I don't see a column for total square feet... am I missing it ?

SFTotals = train_df.filter(regex="SF")
train_df['totalsf'] = SFTotals.sum(axis=1)

# plt.scatter(train_df['totalsf'], train_df['SalePrice'])
# plt.show()

# #Oh yeah, clearly linear.

# plt.scatter(train_df['YearBuilt'], train_df['SalePrice'])
# plt.show()

# #Also linear for the most part, eponential.

# #plt.scatter(train_df['OverallCond'], train_df['SalePrice'])
# #plt.show()

# yes

# #plt.scatter(train_df['OverallCond'], train_df['SalePrice'])
# #plt.show()

# I don't know what the difference is between overall condition and overall quality.
# # For instance...

# plt.scatter(train_df['BsmtQual'], train_df['SalePrice'], c = "orange")
# plt.scatter(train_df['BsmtCond'], train_df['SalePrice'], c = "blue")
# plt.legend()
# plt.show()

# I don't really undersand the difference between those two, but I will use them as they both
# look to have decent purity.

# # We can start with these and see how we do, fortunately, we have already encoded these.

train_new_x = train_df[['totalsf', 'YearBuilt', 'OverallCond', 'BsmtQual', 'BsmtCond']]
dec_clipped = DecisionTreeClassifier(criterion = 'entropy', max_depth = 60, random_state = 99)
dec_cut_fit = dec_clipped.fit(train_new_x, train_df_y)

## I Need a square fottage for test
SFTotals = test.filter(regex="SF")
test['totalsf'] = SFTotals.sum(axis=1)

test_2 = test[['totalsf', 'Year Built', 'Overall Cond', 'Bsmt Qual', 'Bsmt Cond']]

##ugh
test_2['BsmtQual'] = test_2['Bsmt Qual'].astype('category').cat.codes
test_2['BsmtCond'] = test_2['Bsmt Cond'].astype('category').cat.codes
test_2 = test_2.drop(['Bsmt Qual', 'Bsmt Cond'], axis = 1)


cutree_predictions = dec_cut_fit.predict(test_2)
predictions_2_cut = pd.DataFrame(cutree_predictions, columns = ['output'])
print(sqrt(mean_squared_error(values['SalePrice'],predictions_2_cut['output'])))

# #30,956 is the error, which is better, but still not great.
# # YOu can see up above that the depth of the tree is 60 because we have a continous varaiable there are a lot of splits happening. To cut the
# # tree at a low level would be bad in that it would average the prices, For accuracy sake we don't really want to do unlesss we were only
# #concerned with the bin that the final price might fall into.

# # I will probably stop here, 30k is about what we could get the RMSE to. This is not terrible for a decision tree using continous variables.
# #This data is clearly linear, so probably not something we would do on a real world example.

# #Another thing to note is that I did two variations of a decision tree, one using entropy and the other not. Python doesn't exactly have
# # a direct translation.


