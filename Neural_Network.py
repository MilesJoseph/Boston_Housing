from tensorflow.python.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline
import tensorflow as tf
import keras.backend as kb
import keras
from keras.optimizers import SGD
import pandas as pd
import numpy as np




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
SFTotals = data.filter(regex="SF")
data['totalsf'] = SFTotals.sum(axis=1)


data_1 = data[['OverallQual', 'totalsf', 'GarageCars', 'ExterQual', 'TotalBsmtSF', '1stFlrSF', 'KitchenQual', 'SalePrice']]
data_1 = data_1.replace([np.inf, -np.inf], np.nan)
data_1 = data_1.dropna()
print(data_1['OverallQual'].max())
#print(data_1.isnull().any())
data_1 = np.array(data)



from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline
import tensorflow as tf
import keras.backend as kb
import keras
from keras.optimizers import SGD


x = data_1[:,0:7]
y = data_1[:,7]
y=np.reshape(y, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x))
xscale=scaler_x.transform(x)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)

model = Sequential()
model.add(Dense(12, input_dim=7, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history = model.fit(x, y, epochs=150, batch_size=50,  verbose=1)


test_ = pd.read_csv("/Users/milesklingenberg/Documents/UWMSBA/590/Data/test_house-1.csv", dtype=CAT_DTYPES)

SFTotals_test = test_.filter(regex="SF")
test_['totalsf_test'] = SFTotals_test.sum(axis=1)

## we could also do one for total quality

QualTotal_test = test_.filter(regex="Qual")
test_["qualttotal_test"] = QualTotal_test.sum(axis=1)

data_2 = test_[['Overall Qual', 'totalsf_test', 'Garage Cars', 'Exter Qual', 'Total Bsmt SF', '1st Flr SF', 'Kitchen Qual']]
data_2['Kitchen Qual'] = data_2['Kitchen Qual'].astype('category').cat.codes
data_2['Exter Qual'] = data_2['Exter Qual'].astype('category').cat.codes


data_3 = np.array(data_2)
pd.set_option('display.max_columns', 500)

x_test = data_3[:,0:7]
Xnew= scaler_x.transform(x_test)
ynew= model.predict(Xnew)
ynew = scaler_y.inverse_transform(ynew)
Xnew = scaler_x.inverse_transform(Xnew)
