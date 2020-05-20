from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Conv1D, Flatten
import random


random.seed(123)


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


print('tensorflow_version:', tf.__version__)
print('keras_version:', keras.__version__)

# #We know which variables going to use based on our data exploration.

data_1 = data[['OverallQual', 'totalsf', 'ExterQual', 'SalePrice']]
#print(data_1.isnull().any())
##x_train = data_1[['OverallQual', 'totalsf', 'GarageCars', 'ExterQual', 'TotalBsmtSF', '1stFlrSF', 'KitchenQual',]]
##y_train = data_1['SalePrice']

sc = StandardScaler()
train = sc.fit_transform(data_1)

x_train = train[:,0:3]
y_train = train[:,3]

test_ = pd.read_csv("/Users/milesklingenberg/Documents/UWMSBA/590/Data/test_house-1.csv", dtype=CAT_DTYPES)
test_ = pd.DataFrame(test_)

SFTotals_test = test_.filter(regex="SF")
test_['totalsf'] = SFTotals_test.sum(axis=1)

test_ = test_[['Overall Qual', 'totalsf', 'Exter Qual']]

test_['ExterQual'] = test_['Exter Qual'].astype('category').cat.codes
#test_['KitchenQal'] = test_['Kitchen Qual'].astype('category').cat.codes
test_1 = test_.drop(columns = ['Exter Qual'])
#test_1 = test_1.drop(columns = ['Kitchen Qual'])

sc = StandardScaler()
test_ = sc.fit_transform(test_1)

x_test = test_[:,0:3]

def build_model():
  model = keras.Sequential([
    layers.Dense(32, activation='relu', input_dim=3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1,  activation = 'linear'),


  ])

  #optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])
  return model

model = build_model()



EPOCHS = 1000

history = model.fit(
  x_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

plotter.plot({'Basic': history}, metric = "mae")
plt.show()


model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=350)

early_history = model.fit(x_train, y_train,
                    epochs=EPOCHS, validation_split = 0.2, verbose=0,
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()]
                         )

plotter.plot({'Basic': early_history}, metric = "mae")

test_predictions = model.predict(x_test)

predictions = pd.DataFrame(test_predictions, columns = ['predictions'])

values = pd.read_excel("/Users/milesklingenberg/Documents/UWMSBA/590/Data/Values.xlsx")
values = sc.fit_transform(values)
values = pd.DataFrame(values, columns = ['SalePrice'])

values = pd.read_excel("/Users/milesklingenberg/Documents/UWMSBA/590/Data/Values.xlsx")
values = sc.fit_transform(values)
values = pd.DataFrame(values, columns = ['SalePrice'])

from sklearn.metrics import mean_squared_error
from math import sqrt

results= pd.concat([predictions, values], axis = 1)
results = results.dropna()
results_untransform = pd.DataFrame(sc.inverse_transform(results[['SalePrice', 'predictions']]), columns = ['SalePrice', 'Predictions'])

a = plt.axes(aspect='equal')
plt.scatter(results_untransform['Predictions'], results_untransform['SalePrice'])
plt.xlabel('True Values [price]')
plt.ylabel('Predictions [price]')
lims = [0, 50]

print(results_untransform)

print(sqrt(mean_squared_error(results_untransform['SalePrice'], results_untransform['Predictions'])))

# #28k, not great.

results_untransform['error'] = results_untransform['SalePrice']-results_untransform['Predictions']

print(results_untransform.to_string())

plt.scatter(results_untransform['error'], results_untransform['SalePrice'])
plt.show()


