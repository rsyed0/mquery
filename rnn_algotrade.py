# using https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/

from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

from pyalgotrade import strategy, plotter
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.technical import ma, rsi, macd, bollinger

from pyalgotrade.bar import Frequency
from pyalgotrade.barfeed.csvfeed import GenericBarFeed

import pandas as pd

max_spend = 0.25
window_size = 10

"""class RNNStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, hidden_units, dense_units, input_shape, activation, verbose=False, is_train_feed=True):
        super(MyStrategy, self).__init__(feed, 10000)
        self.__instrument = instrument

        # TODO use RNN with onBars methods as inputs, -1 to 1 output
        # TODO use RNN with direct prices as inputs, -1 to 1 output
        self.__model = Sequential()
        self.__model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
        self.__model.add(Dense(units=dense_units, activation=activation[1]))
        self.__model.compile(loss='mean_squared_error', optimizer='adam')
        self.__model.summary()

        self.__cbasis = 0

        self.__thisBatch = np.array([])

        self.__lastTop, self.__lastBottom, self.__trend, self.__trendLength = None, None, 0, 0
        self.__portValues, self.__cprices, self.__shareValues = [], [], []
        self.__maxSpend = max_spend

        self.__sma = ma.SMA(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__rsi = rsi.RSI(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__macd = macd.MACD(feed[instrument].getPriceDataSeries(), fastEmaPeriod, slowEmaPeriod, signalEmaPeriod)
        self.__bb = bollinger.BollingerBands(feed[instrument].getPriceDataSeries(), bBandsPeriod, 2)

        self.__verbose = verbose
        self.__isTrainFeed = is_train_feed

    def set_position(self, position):
        self.__position = position

    def get_position(self):
        return self.__position

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

    def set_last_top(self, lt):
        self.__lastTop = lt

    def set_last_bottom(self, lb):
        self.__lastBottom = lb

    def set_trend(self, t):
        self.__trend = t

    def set_trend_length(self, tl):
        self.__trendLength = tl

    def get_model(self):
        return self.__model

    def onBars(self, bars):
        # TODO train/run self.__model with c_price
        n_shares = self.getBroker().getShares(self.get_instrument())
        bar = bars[self.get_instrument()]
        c_price = bar.getPrice()
        strat_cash = self.getBroker().getCash(False)

        self.__thisBatch = np.append(self.__thisBatch, c_price)
        if len(self.__thisBatch) == window_size:
            # TODO set delta_shares by running self.__model
            # TODO set up model to predict next day's % change, map to delta_shares
            if self.__isTrainFeed:
                # train using self.__model.fit(x,y)
                self.__model.fit(self.__cprices[-window_size:], self.__thisBatch)
            else:
                # run regularly using self.__model.predict()
                result = self.__model.predict(self.__thisBatch)
                print(result)

        if not self.__isTrainFeed:
        # prevent short selling
            if delta_shares < 0 and n_shares < -delta_shares:
                delta_shares = -n_shares

            if self.__verbose:
                print("Day ",len(self.__portValues),": have",n_shares,"shares and $",strat_cash, end="")
                delta_shares = int(delta_shares)

                if delta_shares > 0:
                    print(", buying",delta_shares,"shares at",c_price)
                elif delta_shares < 0:
                    print(", selling",abs(delta_shares),"shares at",c_price)
                else:
                    print("")
                #print(gfill_onBars(self, bars))

            #print([st_onBars(self, bars) for st_onBars in self.__onBars])

            if not delta_shares == 0:
                self.marketOrder(self.__instrument, delta_shares)
                self.__cbasis = (n_shares*self.__cbasis + delta_shares*c_price) / (n_shares + delta_shares) if not n_shares + delta_shares == 0 else 0
        self.__portValues.append(self.getBroker().getEquity())
        self.__cprices.append(c_price)
        self.__shareValues.append(n_shares*c_price)"""

# TODO adapt for indicator inputs instead
def csv_to_xy(path):
    csv_df = pd.read_csv(path)
    cprices = list(csv_df["Close"])

    pct_chgs = [(cprices[i+1]-cprices[i]) / cprices[i] for i in range(len(cprices)-1)]

    # use percent changes in a given window as input x
    X_train = np.array([np.array(pct_chgs[i:i+window_size]) for i in range(len(cprices)-window_size)])

    # use percent change in next interval as output y
    y_train = np.array(pct_chgs[window_size:])

    return X_train, y_train

def main():
    # Load the bar feed from the CSV file
    stock = sys.argv[1].lower()
    period = sys.argv[2].lower() if len(sys.argv) >= 3 else "1y"
    interval = sys.argv[3].lower() if len(sys.argv) >= 4 else ""

    # TODO segment into testing and training feeds
    # TODO use pandas to read csv to get prices before training step
    path = ""
    
    #train_feed = quandlfeed.Feed() if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
    if len(interval) == 0:
        path = "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower())
    else:
        path = "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), period.lower(), interval.lower())
    X_train, y_train = csv_to_xy(path)

    # TODO train RNN outside strategy framework, then build strategy for testing side
    # TODO figure out how to use indicator values in training later on
    model = Sequential()
    model.add(SimpleRNN(2, input_shape=(window_size, 1), activation='linear'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    model.fit(X_train, y_train, epochs=10)

    #rnn_strategy = RNNStrategy(train_feed, stock, 2, 1, (window_size,1), activation=['linear', 'linear'])
    #rnn_strategy.run()

    print("Final portfolio value: $%.2f" % myStrategy.getBroker().getEquity())
    print(rnn_strategy.get_model().summary())
