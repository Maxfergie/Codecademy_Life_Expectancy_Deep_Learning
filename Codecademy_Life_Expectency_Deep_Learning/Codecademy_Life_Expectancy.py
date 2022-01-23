import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

dataset = pd.read_csv('life_expectancy.csv')
#print(dataset.describe())

dataset = dataset.drop(['Country'], axis=1)
labels = dataset.iloc[:,-1]

#PREPROCESSING
#one-hot-encoding for categorical columns
features = pd.get_dummies(dataset)
#split data into training and testing batches
features_train, features_test, labels_train, labels_test =  train_test_split(features,
                                labels,
                                test_size=.33,
                                random_state=42)
#define numerical data
numerical_feats = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_feats.columns
#standardize data
ct = ColumnTransformer([("only numeric",
                          StandardScaler(),
                          numerical_columns)],
                          remainder='passthrough')
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

#MODEL
#define model used
my_model = Sequential()
#establish and load input layer
input = InputLayer(input_shape=(features.shape[1],))
my_model.add(input)
#add hidden layers and activation function
my_model.add(layers.Dense(128, activation='relu'))
my_model.add(layers.Dropout(0.2))
my_model.add(layers.Dense(64, activation='relu'))
#add output layer
my_model.add(Dense(1))
#print(my_model.summary())

#OPTIMIZE AND COMPILE
opt = Adam(learning_rate=0.01)
my_model.compile(loss='mse',
                 metrics='mae',
                 optimizer=opt)

#FITTING AND EVALUATION
#set up early stopping
stop = EarlyStopping(monitor='loss',
                     mode='min',
                     verbose=1,
                     patience=40)
#fit model
my_model.fit(features_train_scaled,
             labels_train,
             epochs=35,
             batch_size=1,
             verbose=1,
             callbacks=[stop])
#evaluate loss
res_mse, res_mae = my_model.evaluate(features_train_scaled,
                                     labels_train,
                                     verbose=0)


print("--------------------------------")
print("Final Loss calculates to:", res_mse)
print("Final Mean Absolute Error calculates to:", res_mae)
