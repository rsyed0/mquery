# using https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/

from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

class RNNStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, hidden_units, dense_units, input_shape, activation):
        super(MyStrategy, self).__init__(feed, 10000)
        self.__instrument = instrument

        # TODO use RNN with onBars methods as inputs, -1 to 1 output
        # TODO use RNN with direct prices as inputs, -1 to 1 output
        self.__model = Sequential()
        self.__model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0]))
        self.__model.add(Dense(units=dense_units, activation=activation[1]))
        self.__model.compile(loss='mean_squared_error', optimizer='adam')

        self.__lastTop, self.__lastBottom, self.__trend, self.__trendLength = None, None, 0, 0
        self.__portValues, self.__cprices, self.__shareValues = [], [], []
        self.__maxSpend = max_spend

        self.__sma = ma.SMA(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__rsi = rsi.RSI(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__macd = macd.MACD(feed[instrument].getPriceDataSeries(), fastEmaPeriod, slowEmaPeriod, signalEmaPeriod)
        self.__bb = bollinger.BollingerBands(feed[instrument].getPriceDataSeries(), bBandsPeriod, 2)

        self.__verbose = verbose

    def onBars(self, bars):
        # run self.__model with c_price
        n_shares = self.getBroker().getShares(self.get_instrument())
        bar = bars[self.get_instrument()]
        c_price = bar.getPrice()
        strat_cash = self.getBroker().getCash(False)

def main():

    # Load the bar feed from the CSV file
    stock = sys.argv[1].lower()
    period = sys.argv[2].lower() if len(sys.argv) >= 3 else "1y"
    interval = sys.argv[3].lower() if len(sys.argv) >= 4 else ""

    feed = quandlfeed.Feed() if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
    if len(interval) == 0:
        feed.addBarsFromCSV(stock, "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower()))
    else:
        feed.addBarsFromCSV(stock, "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), period.lower(), interval.lower()))

    rnn = RNNStrategy(feed, stock, 2, 1, (1,1), activation=['linear', 'linear'])
