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


##Let's do a little feature engineering, and then we can check how those features perform.

#Here we have total square foot which is a combination of all columns containing "sf"
SFTotals = data.filter(regex="SF")
data['totalsf'] = SFTotals.sum(axis=1)

#Here is a total quality, same as above.
QualTotal = data.filter(regex="Qual")
data["qualttotal"] = QualTotal.sum(axis=1)


##let's look at correlation with the absolute value sorted.
## It is clear that the two create variables we created have actually the highest correlation
##with our dependent variable. So, we know we are on the right track.
df_corr =data[data.columns[1:]].corr()['SalePrice'][:]
df_corr = pd.DataFrame(df_corr)
df_corr['abs_value_corr'] = df_corr['SalePrice'].apply(lambda x: abs(x) )
df_corr = df_corr.sort_values('abs_value_corr', ascending =False)
print(df_corr.head(25))



#let's look at those two
plt.scatter(data['OverallQual'], data['SalePrice'])
plt.show()



##The distribution of our categories makes it appear that linear regression would lead
##to a high error rate.

#let's look at total sf

plt.scatter(data['totalsf'], data['SalePrice'])
plt.show()



#Let's look at some basics of the data, we kind of skipped over that.
#Let's check for nulls and then we will check for min and max.
#print(data.info())
#we have already changed cat = int for hot encoding.
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

missing_values_table(data)



# So this isn't too bad, just three columns.
#None of them are categorical so this is also good, for GarageBuilt I am going to impute 0, because
# most likely it does not have a garage.
#MasVnrArea we can impute mean, it is masonary veneer area in square feet, not really sure what that means
#LotFrontage we will impute mean and it is the area of street that the front of the house oocupies.
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)
data['MasVnrArea'] = data.fillna(data['MasVnrArea'].mean())
data['LotFrontage'] = data.fillna(data['LotFrontage'].mean())


#We could look at the variables with the most signifigance with/for linear rfe.
pd.set_option('display.float_format', lambda x: '%.10f' % x)

x = data.iloc[:, 1:80]
y = data.iloc[:, 80]

cols = list(x.columns)
pmax = 1
while (len(cols) > 0):
    p = []
    X_1 = x[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y, X_1).fit()
    p = pd.Series(model.pvalues.values[1:], index = cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break

selected_features_be = cols
print(p.sort_values(ascending =True))


##Quite a few columns that have signifigance.
<<<<<<< HEAD
##Let's see if we can do some dimensionality reduction.



from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators = 1000, random_state=0, n_jobs=-1)


clf.fit(x,y)
##Looking at gini purity


feat_labels = list(data.iloc[:,0:80])
clf_features = pd.DataFrame(clf.feature_importances_)
feat_labels = pd.DataFrame(feat_labels)
merged_df = [feat_labels, clf_features]
result = pd.concat(merged_df, axis = 1)
result.columns = ['feat', 'importance']
result_1 = result.sort_values(by = 'importance')

results_2 = result_1.head(15)

plt.bar(results_2['feat'], results_2['importance'])
plt.xticks(rotation=75)
plt.show()


#This shows us gini purity, but might not be extremely helpful.
#Because we are doing tree based, there is not a huge point for doing dimensionality reduction


data.to_csv(r'/Users/milesklingenberg/Documents/UWMSBA/590/Data/cleaned_data.csv', index=False)

#There is some interesting stuff to try here. time to move to the modeling. 
=======
##Let's see if we can do some dimensionality reduction. 
>>>>>>> 2fcd2c52e6c19520b1cfd5cdd2bff9620f57edad
