from pyalgotrade import strategy, plotter
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.technical import ma, rsi, macd, bollinger

from pyalgotrade.bar import Frequency
from pyalgotrade.barfeed.csvfeed import GenericBarFeed

from strategies import *
from rnn_algotrade import RNNStrategy

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, InputLayer, LSTM, Dropout
from tensorflow.keras.optimizers import SGD

import sys, os
from os.path import exists

import matplotlib.pyplot as plt

from math import log

MAX_Y = 1
VOL_SYMBOL = 'uvxy'

n_models = 7

n_epochs = 15 # 25 - overtrain?

window_size = 3

# TODO implement
batch_size = 1

total_cash = 100000

dense_units = 128
lstm_units = 256

default_max_spend = 0.25
default_norm_factor = 2

default_dtdb_buy = 0.1
default_dtdb_sell = 0.5

class RiskFactorRNNStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, model, norm_factor=default_norm_factor, smaPeriod=15, fastEmaPeriod=12, slowEmaPeriod=26, signalEmaPeriod=9, bBandsPeriod=1, max_spend=default_max_spend, fourier=False, adj_func=None, dtdb_buy=default_dtdb_buy, dtdb_sell=default_dtdb_sell):
        super(RiskFactorRNNStrategy, self).__init__(feed, total_cash)
        self.__position = None
        self.__instrument = instrument

        self.__model = model

        self.__onBars = [sma_onBars, rsi_onBars, smarsi_onBars, macd_onBars, cbasis_onBars, gfill_onBars, channel_onBars] #history_onBars, energy_onBars]
        self.__cbasis = 0

        self.__normFactor = norm_factor
        self.__adjFunc = adj_func

        # TODO implement this
        self.__dtdbBasis = None

        self.__dtdbBuy, self.__dtdbSell = dtdb_buy, dtdb_sell

        self.__vols = []
        self.__startCprice, self.__startVol = None, None

        self.__xWindow = []

        self.__lastTop, self.__lastBottom, self.__trend, self.__trendLength = None, None, 0, 0
        self.__portValues, self.__cprices, self.__shareValues = [], [], []
        self.__maxSpend = max_spend

        self.__sma = ma.SMA(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__rsi = rsi.RSI(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__macd = macd.MACD(feed[instrument].getPriceDataSeries(), fastEmaPeriod, slowEmaPeriod, signalEmaPeriod)
        self.__bb = bollinger.BollingerBands(feed[instrument].getPriceDataSeries(), bBandsPeriod, 2)

        self.__indValues = []
        self.__window = np.array([])

    def get_model(self):
        return self.__model

    def get_instrument(self):
        return self.__instrument

    def get_sma(self):
        return self.__sma

    def get_rsi(self):
        return self.__rsi

    def get_macd(self):
        return self.__macd

    def get_bb(self):
        return self.__bb

    def get_cbasis(self):
        return self.__cbasis

    def get_port_values(self):
        return self.__portValues

    def get_cprices(self):
        return self.__cprices

    def get_share_values(self):
        return self.__shareValues

    def get_last_top(self):
        return self.__lastTop

    def get_last_bottom(self):
        return self.__lastBottom

    def get_trend(self):
        return self.__trend

    def get_trend_length(self):
        return self.__trendLength

    def get_ind_values(self):
        return np.array(self.__indValues)

    def set_last_top(self, lt):
        self.__lastTop = lt

    def set_last_bottom(self, lb):
        self.__lastBottom = lb

    def set_trend(self, t):
        self.__trend = t

    def set_trend_length(self, tl):
        self.__trendLength = tl

    def onBars(self, bars):
        c_price = bars[self.__instrument].getPrice()
        vol_price = bars[VOL_SYMBOL].getPrice()
        n_shares = self.getBroker().getShares(self.__instrument)
        strat_cash = self.getBroker().getCash(False)

        if len(self.__cprices) == 0:
            self.__startCprice = c_price
            self.__startVol = vol_price

        # normalize test interval using norm_factor
        norm_cprice = (c_price - self.__startCprice) / (self.__normFactor * self.__startCprice - self.__startCprice)
        norm_vol = (vol_price - self.__startVol) / (self.__normFactor * self.__startVol - self.__startVol)

        x = [norm_cprice, norm_vol]
        x.extend([st_onBars(self, bars) for st_onBars in self.__onBars])
        self.__xWindow.append(x)

        if len(self.__xWindow) == window_size:

            days_to_dip_below = self.__adjFunc(self.__model.predict(np.array(self.__xWindow))) if self.__adjFunc else self.__model.predict(np.array(self.__xWindow))

            if days_to_dip_below < self.__dtdbBuy:
                if n_shares > 0:
                    self.marketOrder(self.__instrument, -n_shares)
            elif days_to_dip_below > self.__dtdbSell:
                delta_shares = int(self.__maxSpend * strat_cash / c_price)
                self.marketOrder(self.__instrument, delta_shares)

            self.__xWindow.pop(0)
        
        self.__cprices.append(c_price)
        self.__vols.append(vol_price)
        self.__portValues.append(self.getBroker().getEquity())
        self.__shareValues.append(n_shares*c_price)


def risk_factor_rnn():
    # read command line options
    stock = sys.argv[1].lower()
    period = sys.argv[2].lower() if len(sys.argv) >= 3 else "1y"
    interval = sys.argv[3].lower() if len(sys.argv) >= 4 else ""

    # IDEA use many factors (strategies.py, UVXY/VIX) to predict # of trading periods until stock drops below current level
    # Will be 0 if stock is negative next session, 999999 if stays above for rest of the period
    # Can use genetic algorithms or LSTM RNN

    # BUY when output is big number (stock will stay above for long), SELL when output is small (stock about to go down)

    stock_path = None
    vol_path = None
    if len(interval) == 0:
        stock_path = "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower())
        vol_path = "WIKI-%s-%s-yfinance.csv" % (VOL_SYMBOL.upper(), period.lower())
    else:
        stock_path = "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), period.lower(), interval.lower())
        vol_path = "WIKI-%s-%s-%s-yfinance.csv" % (VOL_SYMBOL.upper(), period.lower(), interval.lower())

    stock_feed = quandlfeed.Feed() if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
    stock_feed.addBarsFromCSV(stock, stock_path)

    vol_feed = quandlfeed.Feed() if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
    vol_feed.addBarsFromCSV(VOL_SYMBOL, vol_path)

    stock_df = pd.read_csv(stock_path)
    vol_df = pd.read_csv(vol_path)

    # use a dummy strategy just to get the values of indicators
    dummy = RNNStrategy(stock_feed, stock, None, is_test=True)
    dummy.run()
    ind_values = dummy.get_ind_values()

    stock_close_prices, vol_close_prices = stock_df['Close'], vol_df['Close']

    y_train = []
    nv = None
    for day_i, stock_close in enumerate(stock_close_prices):
        for day_j in range(day_i+1, len(stock_close_prices)):
            if stock_close_prices[day_j] < stock_close:
                y_train.append(day_j - day_i)
                break
        if len(y_train) == day_i:
            y_train.append(len(stock_close_prices) - day_i)

    # norm method 1
    y_train = [y_val / len(stock_close_prices) for y_val in y_train]

    # norm method 2
    #y_train = [log(y_val) for y_val in y_train]
    #mx_y_train = max(y_train)
    #y_train = [y_val / mx_y_train for y_val in y_train]
    
    #y_train += [MAX_Y for i in range(len(stock_close_prices)-len(y_train)-1)]
    #assert len(y_train) == len(stock_close_prices)

    y_train = np.array(y_train[window_size:-1])
    print(y_train)

    X_train = []
    stock_close_min, stock_close_max = min(stock_close_prices), max(stock_close_prices)
    vol_close_min, vol_close_max = min(vol_close_prices), max(vol_close_prices)

    for stock_close, vol_close, day_ind_values in zip(stock_close_prices, vol_close_prices, ind_values):
        stock_close = 2 * ((stock_close - stock_close_min) / (stock_close_max - stock_close_min)) - 1
        vol_close = 2 * ((vol_close - vol_close_min) / (vol_close_max - vol_close_min)) - 1
        day_X_train = [stock_close, vol_close]
        day_X_train.extend(day_ind_values)
        X_train.append(day_X_train)

    #assert len(X_train) == len(stock_close_prices)

    X_train = np.array([X_train[i:i+window_size] for i in range(len(X_train)-window_size)])
    X_train = np.reshape(X_train, (len(X_train), window_size, n_models+2)) # len(stock_close_prices)-window_size
    #print(X_train)

    if len(X_train) > len(y_train):
        X_train = X_train[:-1]

    # build model
    model = Sequential()

    # TODO decide whether to use window
    model.add(LSTM(lstm_units, input_shape=(window_size, n_models+2, ), activation='linear'))
    #model.add(SimpleRNN(lstm_units, input_shape=(window_size, n_models+2, ), activation='linear'))
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

    y_hat = []
    for day in range(len(X_train)):
        """stock_close, vol_close, day_ind_values = stock_close_prices[day], vol_close_prices[day], ind_values[day]
        day_X_train = [stock_close, vol_close]
        day_X_train.extend(day_ind_values)"""
        day_X_train = X_train[day]
        day_X_train = np.reshape(day_X_train, (1, window_size, n_models+2))
        y_hat.append(model.predict(day_X_train)[0][0])
    y_hat = np.array(y_hat)

    plt.hist(y_train)
    plt.show()

    plt.hist(y_hat)
    plt.show()

    plt.scatter(y_hat,y_train)
    plt.show()

    norm_prices = stock_df['Close'][window_size:len(y_train)+window_size]
    mn_price, mx_price = min(norm_prices), max(norm_prices)
    norm_prices = [(x - mn_price) / (mx_price - mn_price) for x in norm_prices]

    plt.plot([i for i in range(len(y_train))], y_train)
    plt.plot([i for i in range(len(y_train))], y_hat)
    plt.plot([i for i in range(len(y_train))], norm_prices)
    plt.show()

    save_i = 1
    while exists("%s-%s-vol-%d.h5" % (stock, period, save_i)):
        save_i += 1    
    model.save("%s-%s-vol-%d.h5" % (stock, period, save_i))

    # refresh feeds
    strat_feed = quandlfeed.Feed() if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
    strat_feed.addBarsFromCSV(stock, stock_path)
    strat_feed.addBarsFromCSV(VOL_SYMBOL, vol_path)

    strat = RiskFactorRNNStrategy(strat_feed, stock, model)
    strat.run()

    final_port_value = strat.getBroker().getEquity()
    print("Final portfolio value: $%.2f" % final_port_value)

    plt.title(stock.upper()+" using Volatility-Based RNN Strategy")
    plt.plot([i for i in range(len(strat.get_port_values()))], strat.get_port_values(), label="Port Value")
    plt.plot([i for i in range(len(strat.get_port_values()))], [p*int(total_cash/strat.get_cprices()[0]) for p in strat.get_cprices()], label="Adj Share Price")
    plt.plot([i for i in range(len(strat.get_port_values()))], strat.get_share_values(), label="Port Value in Shares")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    risk_factor_rnn()