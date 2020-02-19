import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, ModelCheckpoint
from keras.optimizers import Adam
import time
from sklearn import preprocessing
import os
from binance.client import Client
import sklearn.metrics as skll

os.chdir('C:/Users/HGuessous/PycharmProjects/LSTM_Currency_Pairs')


api_key = 'Binance API Key'
api_secret = 'Binance API Key'

#Connect to Binance Client
client = Client(api_key, api_secret)

#Get historical prices and transform into 10 min candles
def get_prices(symbol):
    df = client.get_historical_klines(symbol=symbol, interval='5m', start_str='6 months ago UTC')
    df = pd.DataFrame(df)
    df.columns = ['Open time', 'Open', 'High', 'Low', ' Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'taker buy base asset volume', 'Taker buy quote asset volume', 'ignore']
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
    df = df[['Open time', 'Open', 'High', 'Low', ' Close', 'Volume']]


    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = df.set_index('time')

    df['open'] = df['open'].astype(float)
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)

    df = (df.resample('10T')
         .agg({'open': 'first', 'close': 'last',
               'high': np.max, 'low': np.min,
               'volume': np.sum}))

    df = df.dropna()
    return(df)



sequence_len = 60  # how long of a preceeding sequence to collect for RNN
future_period = 12  # how far into the future are we trying to predict?
target_pair = "BTCUSDT"
epochs = 1  
batch = 64  #
name = f"{sequence_len}-SEQ-{future_period}-PRED-{int(time.time())}"

#Classify buy or sell signals
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


#Normalise data for training
def preprocess_df(df):
    df = df.drop("future", 1)  # don't need this anymore.
    df = df[(df.loc[:, df.columns != 'target'] != 0).all(1)]
    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.


    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=sequence_len)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == sequence_len:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!


    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!


main_df = pd.DataFrame() # begin empty

ratios = ["BTCUSDT"]
for ratio in ratios:

    print(ratio)
    df = get_prices(ratio)  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    #df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
    #df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

#main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)
#print(main_df.head())  # how did we do??

main_df['future'] = main_df[f'close'].shift(-future_period)
main_df['target'] = list(map(classify, main_df[f'close'], main_df['future']))

main_df.dropna(inplace=True)


pred_x, pred_y = preprocess_df(main_df)


model = Sequential()
model.add(LSTM(128, input_shape=(pred_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

model.load_weights('60-SEQ-12-PRED-1573117504')

# Score model and calculate theoretical profits
prediction = model.predict(pred_x, verbose=0)
prediction = pd.DataFrame(prediction)
prediction['act'] = pred_y

prediction1 = main_df[len(main_df) - len(prediction):len(main_df)][['close', 'target']]
prediction1.columns = ['close', 'act']
prediction1['prob'] = prediction[1].values
prediction1['pred'] = [round(value) for value in prediction1['prob']]
prediction1['close1'] = prediction1['close'].shift(periods=-1)
prediction1['change'] = prediction1['close1']/prediction1['close']
prediction1['position'] =  np.where(prediction1['prob'] >= .56,1,
                                np.where(prediction1['prob'] < .38 ,-1,0))

prediction1['position'] = prediction1['position'].replace(0,np.nan).ffill().replace(np.nan,0)
prediction1['trade'] = np.where(prediction1['position'] != prediction1['position'].shift(1),1,0)
prediction1['gain'] = np.where(prediction1['position']==1,(prediction1['change']),(1/(((prediction1['change']-1))+1)))
#prediction1['gain'] = np.where(prediction1['position']==1,(prediction1['change']),1)
prediction1['gain2'] = np.where(prediction1['trade'] ==1,prediction1['gain']-0.001 ,prediction1['gain'] )
prediction1['cum_gain'] = np.cumprod(prediction1['gain2'])

prediction1['position'].mean()

accuracy = skll.accuracy_score(prediction1.loc[:,'act'], prediction1.loc[:,'pred'])
kappa = skll.cohen_kappa_score(prediction1.loc[:,'act'], prediction1.loc[:,'pred'])
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(kappa)
