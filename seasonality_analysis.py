import pandas as pd
import numpy as np

from datetime import datetime

from math import cos, sqrt

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, InputLayer, LSTM, Dropout
from tensorflow.keras.optimizers import SGD

import sys, os

window_size = 5
dense_units = 128
lstm_units = 256
n_epochs = 10
daily_pct_max = 0.1

assert window_size >= 1

def main():
    # read command line options
    stock = sys.argv[1].lower()
    period = sys.argv[2].lower() if len(sys.argv) >= 3 else "1y"
    interval = sys.argv[3].lower() if len(sys.argv) >= 4 else ""

    stock_path = None
    if len(interval) == 0:
        stock_path = "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower())
    else:
        stock_path = "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), period.lower(), interval.lower())

    # PT 1: use day of week as input feature
    stock_df = pd.read_csv(stock_path)
    stock_dates = [[int(s) for s in date.split("-")] for date in stock_df['Date']]
    stock_dates = [datetime(x[0],x[1],x[2]) for x in stock_dates]
    stock_dayofweek = [x.weekday() / 7 for x in stock_dates]

    # PT 2: use seasonality as input feature
    stock_seasonality = [cos(x.timetuple().tm_yday * 6.28 / 366) / 2 + 0.5 for x in stock_dates]

    # PT 3: use previous (window_size) normalized values
    stock_prices = stock_df['Close']
    mn_sp, mx_sp = min(stock_prices), max(stock_prices)
    norm_stock_prices = [(x-mn_sp) / (mx_sp-mn_sp) for x in stock_prices]

    # PT ...: additional data points
    # TODO use dropout nearer to input layer to directly screen out useless factors?

    X_train = np.array([[stock_dayofweek[i], stock_seasonality[i]] + norm_stock_prices[i:i+window_size] for i in range(len(norm_stock_prices)-window_size)])

    # GOAL: use next normalized price values as output value
    # RNN predicts next price, strategy compares prediction to last observed value
    # TODO decide if data needs to be de-normalized (ie, 1y vs 5y)

    y_train = np.array(norm_stock_prices[window_size:])

    # build model
    model = Sequential()

    # TODO decide whether to use LSTM
    model.add(LSTM(lstm_units, input_shape=(window_size+2, ), activation='linear'))
    model.add(Dense(units=dense_units, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=dense_units//2, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='tanh'))

    model.summary()

    # train model
    opt = SGD(learning_rate=0.05, momentum=0.1)
    model.compile(loss='mean_squared_error', optimizer=opt)

    model.fit(X_train, y_train, epochs=n_epochs)

    buy_sell_rtgs = []

    for row in X_train:
        res = model.predict(row)
        buy_sell_rtgs.append((res - row[-1]) / daily_pct_max)

    plt.plot([i for i in range(len(X_train))], norm_stock_prices[window_size-1:-1])
    plt.plot([i for i in range(len(X_train))], buy_sell_rtgs)
    plt.show()

if __name__ == "__main__":
    main()