import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

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


    return data

data = create_data_categorical(data)


# # we can start by looking at some of the data intuitively.

#for col in data:
    #print(col)

# Let's look at correlation first.

# corr = data.corr()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,len(data.columns),1)
# ax.set_xticks(ticks)
# plt.xticks(rotation=90)
# ax.set_yticks(ticks)
# ax.set_xticklabels(data.columns)
# ax.set_yticklabels(data.columns)
# plt.show()

# let's just look at correlation and sort it

# #coming back, let's recreate the total square footage.

SFTotals = data.filter(regex="SF")
data['totalsf'] = SFTotals.sum(axis=1)

# #we could also do one for total quality

QualTotal = data.filter(regex="Qual")
data["qualttotal"] = QualTotal.sum(axis=1)

df_corr =data[data.columns[1:]].corr()['SalePrice'][:]
df_corr = pd.DataFrame(df_corr)
df_corr['abs_value_corr'] = df_corr['SalePrice'].apply(lambda x: abs(x) )
df_corr = df_corr.sort_values('abs_value_corr', ascending =False)
#print(df_corr.head(25))

# plt.scatter(data['OverallQual'], data['SalePrice'])
# plt.show()

# plt.scatter(data['totalsf'], data['SalePrice'])
# plt.show()

# plt.scatter(data['ExterQual'], data['SalePrice'])
# plt.show()

# plt.scatter(data['GarageArea'], data['SalePrice'])
# plt.show()

# Really good correlation.

# plt.bar(data['ExterQual'], data['SalePrice'])
# plt.show()

# #strange

# plt.bar(data['KitchenQual'], data['SalePrice'])
# plt.show()


list_variables_x = list(data[['OverallQual', 'totalsf', 'GrLivArea', 'GarageCars', 'ExterQual', 'GarageArea', 'TotalBsmtSF', '1stFlrSF',\
                       'KitchenQual', 'FullBath', 'TotRmsAbvGrd']])

x= data[['OverallQual', 'totalsf', 'GrLivArea', 'GarageCars', 'ExterQual', 'GarageArea', 'TotalBsmtSF', '1stFlrSF',\
                       'KitchenQual', 'FullBath', 'TotRmsAbvGrd']]

y_var = data['SalePrice']

x_1 = sm.add_constant((x))
model = sm.OLS(y_var, x_1).fit()

cols = list(x.columns)
pmax = 1
while (len(cols) > 0):
    p = []
    X_1 = x[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y_var, X_1).fit()
    p = pd.Series(model.pvalues.values[1:], index = cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

selected_features_be = cols
print(selected_features_be)

# #Below are the columns to keep based on the contribution to the model.

