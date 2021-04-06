import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pyplot 
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from pandas import read_csv


#Load Bitcoin price from yahoo finance
end = datetime.today()
start = datetime(end.year-5,end.month,end.day)
BTC = pdr.DataReader('BTC-USD','yahoo',start,end)
#BTC.to_csv('BTC_USD.csv',index=False)

#read Google trend data  
df=pd.read_csv('multiTimeline.csv',index_col=[0], parse_dates=True)
df1=df.rename(columns={"bit": "bitcoin"})

#Resampling , merge data and sort
bt=BTC['Close'].resample('W').last()
df1=df1.resample('W').sum()
st=pd.merge(df1, bt, left_index=True, right_index=True, how='outer')
st=st.sort_index().dropna()


#Plot raw data 
st.plot(figsize=(15,4))
st.plot(subplots=True, figsize=(15,6))
st.plot(y=["R", "F10.7"], figsize=(15,4))
st.plot(x="R", y=["F10.7", "Dst"], style='.')

# plot cross corelation plot 
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.xcorr(st.Close, st.bitcoin, usevlines=True,
          maxlags=10, normed=True,
          lw=2)
ax1.grid(True)
ax1.axhline(0, color='blue', lw=2)
plt.show()

st.reset_index()
values = st.values
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# Here is created input columns which are (t-n, ... t-1)
	for i in range(n_in, 0, -2):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# Here is created output/forecast column which are (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# here checked values numeric format 
values = values.astype('float32')
print(values)

# Dataset values are normalized by using MinMax method
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
#print(scaled)

# Normalized values are converted for supervised learning 
reframed = series_to_supervised(scaled,1,1)
#reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())

# Dataset is splitted into two groups which are train and test sets
values = reframed.values 
train_size = int(len(values)*0.70)
train = values[:train_size,:]
test = values[train_size:,:]

# Splitted datasets are splitted to trainX, trainY, testX and testY 
trainX, trainY = train[:,:-1], train[:,-1]
testX, testY = test[:,:-1], test[:,-1]
print(trainY, trainY.shape)

# Train and Test datasets are reshaped in 3D size to be used in LSTM
trainX = trainX.reshape((trainX.shape[0],1,trainX.shape[1]))
testX = testX.reshape((testX.shape[0],1,testX.shape[1]))
print(trainX.shape, trainY.shape,testX.shape,testY.shape)

# LSTM model is created and adjusted neuron structure
model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.05))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mae', optimizer='adam')
# Dataset is trained by using trainX and trainY
history = model.fit(trainX, trainY, epochs=20, batch_size=25, validation_data=(testX, testY), verbose=2, shuffle=False)

# Loss values are calculated for every training epoch and are visualized
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.title("Test and Train set Loss Value Rate")
pyplot.xlabel('Epochs Number', fontsize=12)
pyplot.ylabel('Loss Value', fontsize=12)
pyplot.legend()
pyplot.show()

# Prediction process is performed for train dataset
trainPredict = model.predict(trainX)
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))
print(trainX.shape)

# Prediction process is performed for test dataset
testPredict = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))
print(testX.shape)

# Trains dataset inverts scaling for training
trainPredict = concatenate((trainPredict, trainX[:, -1:]), axis=1)
trainPredict = scaler.inverse_transform(trainPredict)
trainPredict = trainPredict[:,0]
print(trainPredict)
print(len(trainPredict))

# Test dataset inverts scaling for forecasting
testPredict = concatenate((testPredict, testX[:, -1:]), axis=1)
testPredict = scaler.inverse_transform(testPredict)
testPredict = testPredict[:,0]

# invert scaling for actual
testY = testY.reshape((len(testY), 1))
inv_y = concatenate((testY, testX[:, -1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
#print('actual: ', len(inv_y))

# Performance measure calculated by using mean_squared_error for train and test prediction
rmse2 = sqrt(mean_squared_error(trainY, trainPredict))
print('Train RMSE: %.3f' % rmse2)
rmse = sqrt(mean_squared_error(inv_y, testPredict))
print('Test RMSE: %.3f' % rmse)

#print(testPredict)
#print(type(trainPredict))

# train and test prediction are concatenated
final = np.append(trainPredict, testPredict)
#print(len(son))

final = pd.DataFrame(data=final, columns=['Close'])
actual = st.Close
actual = actual.values
actual = pd.DataFrame(data=actual, columns=['Close'])

# Finally training and prediction result are visualized
pyplot.plot(actual.Close, 'b', label='Original Set')
pyplot.plot(final.Close[0:182], 'r' , label='Training set')
pyplot.plot(final.Close[182:len(final)], 'g' , label='Predicted/Test set')
pyplot.xlabel('Weekly Time', fontsize=12)
pyplot.ylabel('Close Price', fontsize=12)
pyplot.legend(loc='best')
pyplot.show()
