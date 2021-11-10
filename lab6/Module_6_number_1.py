#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
#%% Read CSV
#First we read the csv to plot the daily confirmed cases
df_3_months = pd.read_csv("california-history.csv",skiprows=range(1,120),nrows=92)
#%% Select Data
#Selecting only the columns for date and US case values
df_3_months = df_3_months[['date', 'positiveIncrease']]
print('\n')
print(df_3_months)

#%%Plotting our first 3 months of data from 3/01/2020 to 5/30/2020
plt.plot(df_3_months['positiveIncrease'])
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')
plt.title('Confirmed Covid-19 Cases in the US (March to May)')
plt.xticks(rotation=70)
plt.show()

#%% Second we read the csv and use it to create our training sets
df = pd.read_csv("california-history.csv",skiprows=range(1,120),nrows=184)

#%%creating a dataframe with only the date and US data
df2 = df[['date', 'positiveIncrease']]
print('\n')
print(df2)

#%%Creating a dataframe with only US data
data = df2.filter(['positiveIncrease'])

#%%converting the dataframe to numpy array
dataset = data.values

#%%Establishing size of training data set which will be the first 3 months previously plotted
#or half (.5) of the total 6 month data
training_data_len = math.ceil(len(dataset)*.5)

#Scaling training set data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#%%Creating training data set
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(90, len(train_data)):
    x_train.append(train_data[i-90:i,0])
    y_train.append(train_data[i,0])

#%%Converting training data sets to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

#Reshaping training set 
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Creating LSTM model
model=Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compiling model
model.compile(optimizer='adam', loss='mean_squared_error')

#Training the model with our training sets and allowing for single iteration
model.fit(x_train, y_train, batch_size=1, epochs=5)

#Creating array for the remaining 3 months of values
test_data = scaled_data[training_data_len-90:, :]

#Creating test data sets
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(90, len(test_data)):
    x_test.append(test_data[i-90: i,0])

#Converting to numpy arrays    
x_test = np.array(x_test)

#Reshaping test data sets
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

#Acquiring predicted values based on test data set
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Plotting trained, validated, and finally predicted data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions']=predictions
plt.title('Confirmed Covid-19 Cases in the US (May to September)')
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')
plt.plot(train['positiveIncrease'])
plt.plot(valid[['positiveIncrease','Predictions']])
plt.legend(['Original', 'Reality', 'Prediction'])
plt.show()