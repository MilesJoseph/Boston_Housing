import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# +
##Let's read in the data. 

data = pd.read_csv("/Users/milesklingenberg/Documents/UWMSBA/590/Data/train_house-1.csv", dtype=CAT_DTYPES)

# +
#We also need to read in and clean the test
# -

test = pd.read_csv("/Users/milesklingenberg/Documents/UWMSBA/590/Data/test.csv", dtype=CAT_DTYPES)

data.head(10)
##Let's take a look at the data. 

# +
sigma = data.SalePrice.std()
mu = data.SalePrice.mean()
med = data.SalePrice.median()
mode = data.SalePrice.mode().to_numpy()

sns.distplot(data.SalePrice)
plt.axvline(mode, linestyle='--', color='green', label='mode')
plt.axvline(med, linestyle='--', color='blue', label='median')
plt.axvline(mu, linestyle='--', color='red', label='mean')
plt.legend()

#here we are just looking at the distribution of the data. 

# +
#Goods news is the data nearly normal distributed. 

# +
fig, axes = plt.subplots(2,2, figsize=(12,12))
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', data=data, ax=axes[0,0])
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallCond', data=data, ax=axes[0,1])
sns.boxplot(x='OverallQual', y='SalePrice', data=data, ax=axes[1,0])
sns.boxplot(x='OverallCond', y='SalePrice', data=data, ax=axes[1,1])
plt.tight_layout()

##In these graphs we are looking at Sale Price compared to sf of general living area. 
##We are using overall quality for hue. 
##In the boxplots we are looking at the Sale price and the distribution by Overall Condition.

# +
##This is a function for looking at the missing data. 

def viz_missing(df):

    missing = pd.DataFrame({
        'Missing':df.isnull().sum(),
        '% Missing': df.isnull().sum()/len(df)
    })
    missing = missing[missing['% Missing'] > 0].sort_values(by='Missing', ascending=False)
    sns.barplot(x=missing.index, y='% Missing', data=missing)
    plt.xticks(rotation=45)
    plt.show()

viz_missing(data)

# +
#Let's look at missing data. 
#We can see that pool qc, fireplace and misc features are some of the highest null values. 
# -

#If we combine, we can see that when Fireplace ==0 and Fireplace is null, it is actually the case 
# of a home not having a fireplace, and such fireplaceQu should be na, not nan.
nofire_na = data.loc[data.FireplaceQu.isna() &
                         (data.Fireplaces == 0)].shape[0]
data.FireplaceQu.isna().sum(), nofire_na


data['FireplaceQu'] = data['FireplaceQu'].cat.add_categories('NA')
data.FireplaceQu.fillna('NA', inplace=True)

# +
#we can check if it is also the case for pool qc as well 
# -

pool_na = data.loc[data.PoolQC.isna() & (data.PoolArea == 0)].shape[0]
pool_na, data.PoolQC.isna().sum()

data['PoolQC'] = data['PoolQC'].cat.add_categories('NA')
data.PoolQC.fillna('NA', inplace=True)

#since we have filled in some of our data, let's take a look again. 
viz_missing(data)

# +
#So you can see that we have definitely gotten rid of some of the missing data pretty easily. 
#Misc Features is tough, so we will move to alley and fence, and replace with median for tolerance to 
#outliers. 
# -

data.MasVnrArea.fillna(data.MasVnrArea.median(), inplace=True)
data.LotFrontage.fillna(data.LotFrontage.median(), inplace=True)
viz_missing(data)

for var in ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']:
    print(data[var].isna().sum())

data['GarageFinish'] = data['GarageFinish'].cat.add_categories('NA')
data['GarageQual'] = data['GarageQual'].cat.add_categories('NA')
data['GarageCond'] = data['GarageCond'].cat.add_categories('NA')

data['GarageType'] = data['GarageType'].cat.add_categories('NA')

for var in ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']:
    data[var].fillna('NA', inplace=True)

# +
#So, for all of the garage variables we have the exact same amoutn of missing variables, which probably means 
#these are homes without garages. We can also check with basement. 
# -

for var in ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']:
    print(data[var].isnull().sum())

# +
#Not quite the same. But, we will fill this in with NA as well. 

# +
data['BsmtExposure'] = data['BsmtExposure'].cat.add_categories('NA')
data['BsmtFinType1'] = data['BsmtFinType1'].cat.add_categories('NA')
data['BsmtFinType2'] = data['BsmtFinType2'].cat.add_categories('NA')
data['BsmtQual'] = data['BsmtQual'].cat.add_categories('NA')
data['BsmtCond'] = data['BsmtCond'].cat.add_categories('NA')

for var in ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']:
    data[var].fillna('NA', inplace=True)
# -

data[['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']].isna().sum()

##forgot to fill miscfeature
data['MiscFeature'] = data['MiscFeature'].cat.add_categories('NA')
data.MiscFeature.fillna('NA', inplace=True)

# +
#Alley and Fence I am going to drop. 
# -

data.loc[data.GarageYrBlt.isna(), 'GarageYrBlt'] = data.YearBuilt
data.MasVnrType.fillna('None', inplace=True) # fill with mode

data['Electrical'] = data['Electrical'].cat.add_categories('Sbrkr')
data.Electrical.fillna('Sbrkr', inplace=True)

data = data.drop(['Alley', 'Fence'], axis=1)

viz_missing(test)

# +
#So, we also need to look at the test data and see if the missing values are consistent with 
#the same missing data in the train data. If the feature does not exist we will insert an NA. 
# -

test['PoolQC'] = test['PoolQC'].cat.add_categories('NA')

# +
m = test.PoolQC.mode()[0]

test.loc[test.PoolQC.isna() & (test.PoolArea > 0), 'PoolQC'] = m

test.loc[test.PoolQC.isna() & (test.PoolArea == 0), ['PoolQC']] = 'NA'
# -

test['MiscFeature'] = test['MiscFeature'].cat.add_categories('NA')
test['FireplaceQu'] = test['FireplaceQu'].cat.add_categories('NA')

test.MiscFeature.fillna('NA', inplace=True)
test.FireplaceQu.fillna('NA', inplace=True)
test.MasVnrArea.fillna(test.MasVnrArea.median(), inplace=True)
test.LotFrontage.fillna(test.LotFrontage.median(), inplace=True)


# Drop these two columns, just as we did for the training data.
test.drop(['Alley', 'Fence'], axis=1, inplace=True)

viz_missing(test)

garage_vals = ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType']
test.loc[test.GarageArea.isna(), garage_vals]

# +
g_type = garage_vals.pop(3)

for g in garage_vals:
    mode = test[g].mode()[0]
    test.loc[test.GarageArea.isna(), g] = mode

med = test.GarageArea.median()
test.loc[test.GarageArea.isna(), 'GarageArea']  = med

garage_vals.append(g_type)    

test.loc[test.GarageArea.isna(), garage_vals]

# +
test['GarageQual'] = test['GarageQual'].cat.add_categories('NA')
test['GarageFinish'] = test['GarageFinish'].cat.add_categories('NA')
test['GarageType'] = test['GarageType'].cat.add_categories('NA')
test['GarageCond'] = test['GarageCond'].cat.add_categories('NA')


for g in garage_vals:
    
    mode = test[g].mode()[0]
    test.loc[test[g].isna() & (test.GarageArea > 0), g] = mode
    test.loc[test[g].isna() & (test.GarageArea == 0), g] = 'NA'
# -

years = test.loc[test.GarageYrBlt.isna(), 'YearBuilt']
test.loc[test.GarageYrBlt.isna(), 'GarageYrBlt'] = years

bsmt_vars = ['BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1']
bsmt_sf = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
test.loc[test.TotalBsmtSF.isna(), bsmt_vars+bsmt_sf]

test.loc[test.TotalBsmtSF.isna(), bsmt_sf] = 0

test.loc[:, bsmt_sf].isna().sum()

# +
test['BsmtExposure'] = test['BsmtExposure'].cat.add_categories('NA')
test['BsmtFinType1'] = test['BsmtFinType1'].cat.add_categories('NA')
test['BsmtFinType2'] = test['BsmtFinType2'].cat.add_categories('NA')
test['BsmtQual'] = test['BsmtQual'].cat.add_categories('NA')
test['BsmtCond'] = test['BsmtCond'].cat.add_categories('NA')

for var in bsmt_vars:
    mode = test[var].mode()[0]
    test.loc[test[var].isnull() & (test.TotalBsmtSF == 0), var] = 'NA'
    test.loc[test[var].isnull() & (test.TotalBsmtSF > 0), var] = mode

    
finish = test.loc[test.BsmtFinType2.isna(), 'BsmtFinType1']
test.loc[test.BsmtFinType2.isna(), 'BsmtFinType2'] = finish
# -

viz_missing(test)

test.loc[test.MasVnrType.isna(), ['MasVnrArea']]

mode = test_df.MasVnrType.mode()[0]
test_df.loc[(test_df.MasVnrType.isna() & (test_df.MasVnrArea > 0)), 'MasVnrType'] = mode

test.loc[test.MasVnrType.isna(), ['MasVnrType']] = 'None'


viz_missing(test)

test.loc[test.MSZoning.isna(), 'Neighborhood']

# +
#What we are doing here is looking at which type of zoning particular neighborhoods have and 
#see if you can impute that for the null values in zoning from the neighborhoods that 
#are shared. 
# -

test.loc[test.Neighborhood == 'IDOTRR', 'MSZoning'].value_counts()
#So all of the null neighborhoods are 'IDOTRR', so we will impute RM for MSZOning Null

#We also need to check for Mitchel 
test.loc[test.Neighborhood == 'Mitchel', 'MSZoning'].value_counts()

# +
#We will impute RL for Mitchel 

mask1 = (test.Neighborhood == 'IDOTRR') & (test.MSZoning.isna())
mask2 = (test.Neighborhood == 'Mitchel') & (test.MSZoning.isna())
test.loc[mask1, 'MSZoning'] = 'RM'
test.loc[mask2, 'MSZoning'] = 'RL'

# -

viz_missing(test)

# Kind of a strange outlier that there is a home with high square footage, high quality that sold below avearge. 

# +
rem_vars = ['Utilities', 'Functional', 'Exterior1st', 
        'Exterior2nd', 'KitchenQual', 'SaleType']

for var in rem_vars:
    mode = test[var].mode()[0]
    test.loc[test[var].isna(), var] = mode
# -

#So this means we still have BsmtFullBath, BsmtHalfBath, and GarageCars left to fix 
viz_missing(test)

test.loc[test.GarageCars.isna(), 'GarageArea']
##For GarageCars we have a garage that is 480 sq. ft. 

# +
test.loc[test.GarageCars.isna(), 'GarageCars'] = 2

##we will assign this as a two car garage because that is a pretty big garage. 
# -

bsmt_bths = ['BsmtFullBath', 'BsmtHalfBath']
mask = (test.BsmtFullBath.isna() | test.BsmtHalfBath.isna())
test.loc[mask, ['BsmtFinSF1']+bsmt_bths]

# +
#Here we are checking if the basment has any finish where the bathroom in basement is null 
# -

test.loc[mask, bsmt_bths] = test.loc[mask, bsmt_bths].fillna(0)

# +
###Let's do some feature engineering and also look at feature importance. 

# +
#Here we have total square foot which is a combination of all columns containing "sf"
SFTotals = data.filter(regex="SF")
data['totalsf'] = SFTotals.sum(axis=1)

#Test
SFTotals = test.filter(regex="SF")
test['totalsf'] = SFTotals.sum(axis=1)

# +
#Here is a total quality, same as above.
QualTotal = data.filter(regex="Qual")
data["qualttotal"] = QualTotal.sum(axis=1)

#Test
#Here is a total quality, same as above.
QualTotal = test.filter(regex="Qual")
test["qualttotal"] = QualTotal.sum(axis=1)


# +
porch_area = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']

data['HasPorch'] = data.loc[:, porch_area] \
                            .sum(axis=1) \
                            .apply(lambda x: 1 if x > 0 else 0)

test['HasPorch'] = test.loc[:, porch_area] \
                            .sum(axis=1) \
                            .apply(lambda x: 1 if x > 0 else 0)

# +
#These are just a few examples of feature engineering that you can perform. 
# -

test.drop(['MiscFeature'], axis=1, inplace=True)
data.drop(['MiscFeature'], axis=1, inplace=True)

#That is where I will stop for the feature engineering, just going to check out a few more 
#visualizations. 
plt.scatter(data['OverallQual'], data['SalePrice'])
plt.show()

# Distribution in categories would probably lead to high error rate with linear. 

#let's look at total sf
plt.scatter(data['totalsf'], data['SalePrice'])
plt.show()

pd.set_option('display.float_format', lambda x: '%.10f' % x)

# +
#Relatioships

#A few things to note, we have three variable types; 
#Continous
#Ordinal
#Nominal

##So what we are doing is segregating the variables by their type. 

exp_ordinal = ['ExterQual', 'BsmtQual', 'GarageQual', 
               'KitchenQual', 'FireplaceQu', 'PoolQC',  
               'OverallQual', 'ExterCond', 'BsmtCond', 
               'GarageCond', 'OverallCond', 'HeatingQC',
               'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
               'GarageCars', 'GarageFinish', 'BedroomAbvGr', 
               'KitchenAbvGr', 'HouseStyle', 'TotRmsAbvGrd',
               'YearBuilt', 'MoSold', 'YrSold', 'MasVnrArea', 
               'Utilities', 'BsmtFullBath', 'BsmtHalfBath', 
               'FullBath', 'HalfBath', 'CentralAir', 
               'YearRemodAdd','LotShape', 'Functional']

exp_nominal = ['Neighborhood', 'MSZoning', 'Condition1', 
               'Condition2', 'RoofStyle', 
               'RoofMatl', 'Exterior1st', 'Exterior2nd', 
               'MasVnrType', 'Foundation', 'Electrical', 
               'SaleType', 'SaleCondition', 'HasShed',
               'HasTennis', 'HasGar2', 'HasPorch', 
               'HasDeck', 'HasPool', 'GarageType', 
               'LotConfig', 'PavedDrive', 'IsOld', 'IsNew']

exp_contin  = ['LotArea', 'LotFrontage', 'GrLivArea', 
               'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2', 
               'TotalBsmtSF', 'BsmtUnfSF', '1stFlrSF', 
               '2ndFlrSF', 'LowQualFinSF', 'WoodDeckSF', 
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
               'PoolArea', 'MiscVal', 'TotIntArea', 
               'SalePrice']



# +
#Foruntaley, the schema for the quality conditions are the same across a large number of variables. 

#Ex = Excellent 
#Gd = Good 
#TA = Avearge/Typical 
#Fa = Fair 
#Po = Poor 

#We also have 'NA' which will we incude as 0, thus 0-5. 

# +
qual_map = {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
qual_vars = ['ExterQual', 'ExterCond', 'BsmtQual', 
             'BsmtCond', 'HeatingQC', 'KitchenQual', 
             'FireplaceQu', 'GarageQual', 'GarageCond',
             'PoolQC']

for col in qual_vars:
    data[col] = data[col].map(qual_map)
    test[col] = test[col].map(qual_map)
    
# Make sure we see all numeric data:    
data[qual_vars]
# -

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


# This shows us gini purity, but might not be extremely helpful.
# Because we are doing tree based, there is not a huge point for doing dimensionality reduction


data.to_csv(r'/Users/milesklingenberg/Documents/UWMSBA/590/Data/cleaned_data.csv', index=False)
