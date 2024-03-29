from pandas import read_csv
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, InputLayer, LSTM, Dropout
#from keras.optimizers import SGD

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import math

import matplotlib.pyplot as plt

import tensorflow as tf

from pyalgotrade import strategy, plotter
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.technical import ma, rsi, macd, bollinger

from pyalgotrade.bar import Frequency
from pyalgotrade.barfeed.csvfeed import GenericBarFeed

from strategies import *

from yfinance_csv_dates import fetch_price_csv
from copy import deepcopy

from random import randint, random

import sys, os

n_survivors = 3
p_mutation = 0.5
msp_dominance = 0.75  # 0.5 to 1

# TODO implement adaptive max_spend
default_max_spend = 0.5 #0.05

pct_adj = 0.1

window_size = 30
fwd_window_size = 20 # too little?

total_cash = 100000
n_models = 7

n_epochs = 10 # 25 - overfit?

#hidden_units = 2
dense_units = 64
lstm_units = 128

fourier_time_intervals = [5, 10, 20] #[5, 10, 15, 25]
fourier_time_weights = [0.5, 0.3, 0.2] #[0.5, 0.25, 0.125, 0.125]

assert len(fourier_time_intervals) == len(fourier_time_weights)

use_ind_values = False
train_msp = True
debug = False

# TODO segment RNN approach into volatility and bullishness
class RNNStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, model, smaPeriod=15, max_spend=default_max_spend, fastEmaPeriod=12, slowEmaPeriod=26, signalEmaPeriod=9, bBandsPeriod=1, is_test=False, verbose=False, fourier=False):
        super(RNNStrategy, self).__init__(feed, total_cash)
        self.__instrument = instrument
        self.__model = model

        self.__onBars = [sma_onBars, rsi_onBars, smarsi_onBars, macd_onBars, cbasis_onBars, gfill_onBars, channel_onBars] #history_onBars, energy_onBars]
        self.__cbasis = 0

        assert n_models == len(self.__onBars)

        self.__lastTop, self.__lastBottom, self.__trend, self.__trendLength = None, None, 0, 0
        self.__portValues, self.__cprices, self.__shareValues = [], [], []
        self.__maxSpend = max_spend

        self.__sma = ma.SMA(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__rsi = rsi.RSI(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__macd = macd.MACD(feed[instrument].getPriceDataSeries(), fastEmaPeriod, slowEmaPeriod, signalEmaPeriod)
        self.__bb = bollinger.BollingerBands(feed[instrument].getPriceDataSeries(), bBandsPeriod, 2)

        self.__verbose = verbose
        self.__isTest = is_test
        self.__fourier = fourier

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
        n_shares = self.getBroker().getShares(self.__instrument)
        bar = bars[self.__instrument]
        c_price = bar.getPrice()
        strat_cash = self.getBroker().getCash(False)

        pd_ind_values = np.array([st_onBars(self, bars) for st_onBars in self.__onBars])
        self.__indValues.append(pd_ind_values)

        if self.__isTest:
            return

        delta_shares = 0
        if use_ind_values:
            if len(self.__indValues) >= window_size:
                x = np.array(self.__indValues[-window_size:])
                x = np.reshape(x, (1, window_size, n_models, 1))
                res = self.__model.predict(x)

                if self.__verbose and debug:
                    print(x,res)

                #print(res)

                if self.__fourier:
                    delta_shares = int(self.__maxSpend*total_cash*sum([wt*r for wt,r in zip(fourier_time_weights,res[0])]))
                else:
                    delta_shares = int(self.__maxSpend*total_cash*res)
            else:
                delta_shares = 0
        else:
            self.__window = np.append(self.__window, c_price)
            if len(self.__window) > window_size:
                x = np.reshape(np.array([(self.__window[i+1] - self.__window[i]) / (self.__window[i] * pct_adj) for i in range(len(self.__window)-1)]), (1, window_size, 1))
                res = self.__model.predict(x)

                if self.__verbose and debug:
                    print(x,res)

                if self.__fourier:
                    delta_shares = int(self.__maxSpend*total_cash*sum([wt*r for wt,r in zip(fourier_time_weights,res[0])]) / c_price)
                else:
                    delta_shares = int(self.__maxSpend*total_cash*res / c_price)
                self.__window = np.delete(self.__window, 0)
            else:
                delta_shares = 0
        
        if delta_shares + n_shares < 0:
            delta_shares = -n_shares

        if delta_shares > 0 and strat_cash < delta_shares*c_price:
            delta_shares = int(strat_cash / c_price)
        
        if self.__verbose:
            print("Day %d: have %d shares and $%-7.2f" % (len(self.__portValues), n_shares, strat_cash), end="")
            if delta_shares > 0:
                print(", buying %d shares at $%-7.2f" % (delta_shares, c_price))
            elif delta_shares < 0:
                print(", selling %d shares at $%-7.2f" % (abs(delta_shares), c_price))
            else:
                print("")
        
        if not delta_shares == 0:
            self.marketOrder(self.__instrument, delta_shares)
            self.__cbasis = (n_shares*self.__cbasis + delta_shares*c_price) / (n_shares + delta_shares) if not n_shares + delta_shares == 0 else 0
        self.__portValues.append(self.getBroker().getEquity())
        self.__cprices.append(c_price)
        self.__shareValues.append(n_shares*c_price)

def normalize(model):
    # make so that all vals sum to 1
    s = sum(model)
    return [x/s for x in model]

# TODO adapt for indicator inputs instead
# TODO implement longer-term results (>1 trading interval)
def csv_to_xy(path):
    csv_df = read_csv(path)
    cprices = list(csv_df["Close"])
    
    pct_chgs = [(cprices[i+1]-cprices[i]) / (cprices[i] * pct_adj) for i in range(len(cprices)-1)]

    # use percent changes in a given window as input x
    X_train = np.reshape(np.array([np.array(pct_chgs[i:i+window_size]) for i in range(len(cprices)-window_size-1)]), (len(cprices)-window_size-1, window_size))

    # use percent change in next interval as output y
    y_train = np.array(pct_chgs)

    # y_t method 1
    # implement longer term bullishness in output
    yt = []
    for i in range(len(pct_chgs)):
        importances, values = [], []
        for j in range(i,min(len(pct_chgs),i+fwd_window_size)):
            importance = fwd_window_size-(j-i)
            importances.append(importance)
            values.append(pct_chgs[j])
        importances = normalize(importances)
        yt.append(sum([imp*val for imp,val in zip(importances, values)]))
    y_train = np.array(yt)
    
    # norm method 1
    #mn,mx = min(y_train), max(y_train)
    #norm_y_train = np.array([2*((x-mn)/(mx-mn))-1 for x in y_train])

    # norm method 2
    #a = sum(y_train) / len(y_train)
    #norm_y_train = np.array([x-a for x in y_train])

    # norm method 3
    mean,sigma = sum(y_train) / len(y_train), np.std(y_train)
    norm_y_train = np.array([(x-mean)/sigma for x in y_train])

    return X_train, y_train, norm_y_train

# use genetic algorithm to tune max_spend parameter
def genetic_train_and_test_rnn(stock, train_start, train_end, test_start, test_end, pop_size=3, n_generations=2):

    train_path = "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), train_start, train_end)
    test_path = "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), test_start, test_end)

    # TODO make this work for intervals != "1d"
    train_feed = quandlfeed.Feed() #if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)

    try:
        train_feed.addBarsFromCSV(stock, train_path)
    except:
        fetch_price_csv(stock.upper(), train_start, train_end)
        train_feed.addBarsFromCSV(stock, train_path)

    X_train, y_train, norm_y_train = csv_to_xy(train_path)

    model = Sequential()
    if use_ind_values:
        test_strat = RNNStrategy(train_feed, stock, None, is_test=True)
        test_strat.run()
        ind_values = test_strat.get_ind_values()[1:]

        X_train = [ind_values[i-window_size:i] for i in range(window_size, len(ind_values))]
        X_train = np.reshape(np.array(X_train), (len(X_train), window_size, n_models, 1))

        # TODO turn this into RNN with n_models inputs per time step, using window_size
        # TODO use separate model to predict max_spend (volatility)

        model.add(LSTM(lstm_units, input_shape=(window_size, n_models,), activation='linear'))
        #model.add(SimpleRNN(hidden_units, input_shape=(window_size, n_models,), activation='linear'))
        #model.add(InputLayer(input_shape=(window_size, n_models,)))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='tanh'))

        #opt = SGD(lr=0.01, momentum=0.9)
        model.compile(loss='mean_squared_error', optimizer='adam')

        if debug:
            model.summary()

        y_train = y_train[window_size:]
        norm_y_train = norm_y_train[window_size:]

        #print(X_train, norm_y_train)

        # TODO go back to y_train ?
        model.fit(X_train, norm_y_train, epochs=n_epochs)
    else:
        # TODO train RNN outside strategy framework, then build strategy for testing side
        # TODO figure out how to use indicator values in training later on

        model.add(LSTM(lstm_units, input_shape=(window_size, 1), activation='linear'))
        #model.add(SimpleRNN(hidden_units, input_shape=(window_size, 1), activation='linear'))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='tanh'))

        #model.compile(loss='mean_squared_error', optimizer='adam')
        #opt = SGD(lr=0.01, momentum=0.9)
        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse'])

        if debug:
            model.summary()

        norm_y_train = norm_y_train[window_size:]

        model.fit(X_train, norm_y_train, epochs=n_epochs)

    genetic_msp = -1
    if train_msp:
        population = [random() for i in range(pop_size)]
        scores = []
        for i in range(n_generations):
            print("Generation %d" % (i+1))
            for max_spend in population:
                #train_feed = quandlfeed.Feed() #if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
                #train_feed.addBarsFromCSV(stock, train_path)

                t_train_feed = deepcopy(train_feed)

                strat = RNNStrategy(t_train_feed, stock, model, max_spend=max_spend, is_test=False)
                strat.run()

                score = strat.getBroker().getEquity()
                scores.append((max_spend, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            print(scores)

            next_pop = []
            for i in range(n_survivors):
                for j in range(i+1, n_survivors):
                    next_pop.append(scores[i][0]*msp_dominance + scores[j][0]*(1-msp_dominance))
                    if len(next_pop) == pop_size:
                        break
                if len(next_pop) == pop_size:
                    break
            population = next_pop

        genetic_msp, train_score = scores[0]
    else:
        genetic_msp = default_max_spend

    print("Max spend parameter: %4.2f" % (genetic_msp))

    test_feed = quandlfeed.Feed() #if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)

    try:
        test_feed.addBarsFromCSV(stock, test_path)
    except:
        fetch_price_csv(stock.upper(), test_start, test_end)
        test_feed.addBarsFromCSV(stock, test_path)

    strat = RNNStrategy(test_feed, stock, model, max_spend=genetic_msp, is_test=False, verbose=True)
    strat.run()

    final_port_value = strat.getBroker().getEquity()
    print("Final portfolio value: $%.2f" % final_port_value)

    plt.title(stock.upper()+" using RNN Strategy")
    plt.plot([i for i in range(len(strat.get_port_values()))], strat.get_port_values(), label="Port Value")
    plt.plot([i for i in range(len(strat.get_port_values()))], [p*int(total_cash/strat.get_cprices()[0]) for p in strat.get_cprices()], label="Adj Share Price")
    plt.plot([i for i in range(len(strat.get_port_values()))], strat.get_share_values(), label="Port Value in Shares")
    plt.legend()
    plt.show()

    if debug:
        plt.plot([i for i in range(len(y_train))], y_train, label="y")
        plt.plot([i for i in range(len(y_train))], norm_y_train, label="norm")
        plt.legend()
        plt.show()

    to_save = input("Save model? (Y or [N]): ")
    if to_save.upper() == "Y":
        this_dir = set(os.listdir())

        i = 1
        model_save_fname = "%s-%4.2f-%d.h5" % (stock, genetic_msp, i)
        while model_save_fname in this_dir:
            i += 1
            model_save_fname = "%s-%4.2f-%d.h5" % (stock, genetic_msp, i)

        model.save(model_save_fname)
        print("Saved model as %s" % (model_save_fname))

def csv_to_xy_fourier(path):
    csv_df = read_csv(path)
    cprices = list(csv_df["Close"])

    black_out = max(fourier_time_intervals)
    x_pct_chgs = [(cprices[i]-cprices[i-1]) / cprices[i-1] for i in range(1,len(cprices))]

    start_i, end_i = window_size, len(cprices)-black_out
    X_train, y_train = [], []
    for i in range(start_i, end_i):
        X_train.append(x_pct_chgs[i-window_size:i])
        y_train.append(np.array([(cprices[i+ti]-cprices[i]) / cprices[i] for ti in fourier_time_intervals]))

        """for j in range(len(fourier_time_intervals)):
            ti = fourier_time_intervals[j]
            y_train[j].append((cprices[i+ti-1]-cprices[i-1]) / cprices[i-1])"""

    # TODO normalize ??

    X_train, y_train = np.array(X_train), np.array(y_train)
    print(X_train.shape, y_train.shape)
    return X_train, y_train

    # use percent changes in a given window as input x
    """X_train = np.reshape(np.array([np.array(x_pct_chgs[i:i+window_size]) for i in range(len(cprices)-black_out-window_size-1)]), (len(cprices)-black_out-window_size-1, window_size))
    print(X_train.shape)
    
    # TODO divide by pct_adj
    pct_chgs = np.array([[(cprices[i+ti]-cprices[i]) / (cprices[i]) for i in range(black_out-ti, len(cprices)-ti)] for ti in fourier_time_intervals])
    print(pct_chgs.shape)

    # TODO use percent changes in a given window as input x
    #X_train = np.reshape(np.array([np.array(pct_chgs[i:i+window_size]) for i in range(len(cprices)-window_size-1)]), (len(cprices)-window_size-1, window_size))

    #X_train = np.array([[pct_chgs[i][j-window_size:j] for j in range(window_size, len(pct_chgs[i]))] for i in range(len(fourier_time_intervals))])
    #print(X_train.shape)

    # TODO normalize pct_chgs (??)

    # use percent change in next interval as output y
    y_train = np.array([pct_chgs[i][fourier_time_intervals[i]:-black_out+fourier_time_intervals[i]-1] for i in range(len(fourier_time_intervals))])
    print(y_train.shape)
    #y_train = np.reshape(y_train, (len(pct_chgs)-black_out, len(fourier_time_intervals)))

    return X_train, y_train"""

def fourier_series_rnn(stock, train_start, train_end, test_start, test_end, pop_size=3, n_generations=2):

    train_path = "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), train_start, train_end)
    test_path = "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), test_start, test_end)

    # TODO make this work for intervals != "1d"
    train_feed = quandlfeed.Feed() #if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)

    try:
        train_feed.addBarsFromCSV(stock, train_path)
    except:
        fetch_price_csv(stock.upper(), train_start, train_end)
        train_feed.addBarsFromCSV(stock, train_path)

    X_train, y_train = csv_to_xy_fourier(train_path)

    output_units = len(fourier_time_intervals)
    black_out = max(fourier_time_intervals)

    model = Sequential()
    if use_ind_values:
        test_strat = RNNStrategy(train_feed, stock, None, is_test=True)
        test_strat.run()
        ind_values = test_strat.get_ind_values()[1:]

        X_train = [ind_values[i-window_size:i] for i in range(window_size, len(ind_values)-black_out+1)]
        X_train = np.reshape(np.array(X_train), (len(X_train), window_size, n_models, 1))

        # TODO turn this into RNN with n_models inputs per time step, using window_size
        # TODO use separate model to predict max_spend (volatility)

        model.add(LSTM(lstm_units, input_shape=(window_size, n_models,), activation='linear'))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=output_units, activation='tanh'))

        #opt = SGD(lr=0.01, momentum=0.9)
        model.compile(loss='mean_squared_error', optimizer='sgd')

        if debug:
            model.summary()

        model.fit(X_train, y_train, epochs=n_epochs)
    else:
        # TODO train RNN outside strategy framework, then build strategy for testing side
        # TODO figure out how to use indicator values in training later on

        model.add(LSTM(lstm_units, input_shape=(window_size, 1), activation='linear'))
        #model.add(SimpleRNN(hidden_units, input_shape=(window_size, 1), activation='linear'))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=output_units, activation='tanh'))

        #model.compile(loss='mean_squared_error', optimizer='adam')
        #opt = SGD(lr=0.01, momentum=0.9)
        model.compile(loss='mean_squared_error', optimizer='sgd')

        if debug:
            model.summary()

        #norm_y_train = norm_y_train[window_size:]

        # TODO go back to norm_y_train ??
        model.fit(X_train, y_train, epochs=n_epochs)

    test_feed = quandlfeed.Feed() #if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)

    try:
        test_feed.addBarsFromCSV(stock, test_path)
    except:
        fetch_price_csv(stock.upper(), test_start, test_end)
        test_feed.addBarsFromCSV(stock, test_path)

    genetic_msp = -1
    if train_msp:
        population = [random() for i in range(pop_size)]
        scores = []
        for i in range(n_generations):
            print("Generation %d" % (i+1))
            for max_spend in population:
                print("Trying msp=%4.2f" % (max_spend))
                #t_train_feed = quandlfeed.Feed() #if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
                #t_train_feed.addBarsFromCSV(stock, test_path)

                t_train_feed = deepcopy(test_feed)

                strat = RNNStrategy(t_train_feed, stock, model, max_spend=max_spend, is_test=False, fourier=True)
                strat.run()

                score = strat.getBroker().getEquity()
                scores.append((max_spend, score))

            scores.sort(key=lambda x: x[1], reverse=True)
            print(scores)

            next_pop = []
            for i in range(n_survivors):
                for j in range(i+1, n_survivors):
                    next_pop.append(scores[i][0]*msp_dominance + scores[j][0]*(1-msp_dominance))
                    if len(next_pop) == pop_size:
                        break
                if len(next_pop) == pop_size:
                    break
            population = next_pop

        genetic_msp, train_score = scores[0]
    else:
        genetic_msp = default_max_spend

    strat = RNNStrategy(test_feed, stock, model, max_spend=genetic_msp, is_test=False, verbose=True, fourier=True)
    strat.run()

    final_port_value = strat.getBroker().getEquity()
    print("Final portfolio value: $%.2f" % final_port_value)

    plt.title(stock.upper()+" using Fourier RNN Strategy with "+str(fourier_time_weights)+", "+str(fourier_time_intervals))
    plt.plot([i for i in range(len(strat.get_port_values()))], strat.get_port_values(), label="Port Value")
    plt.plot([i for i in range(len(strat.get_port_values()))], [p*int(total_cash/strat.get_cprices()[0]) for p in strat.get_cprices()], label="Adj Share Price")
    plt.plot([i for i in range(len(strat.get_port_values()))], strat.get_share_values(), label="Port Value in Shares")
    plt.legend()
    plt.show()

    if debug:
        plt.plot([i for i in range(len(y_train))], y_train, label="y")
        plt.plot([i for i in range(len(y_train))], norm_y_train, label="norm")
        plt.legend()
        plt.show()

    to_save = input("Save model? (Y or [N]): ")
    if to_save.upper() == "Y":
        this_dir = set(os.listdir())

        i = 1
        model_save_fname = "%s-%4.2f-%d-fourier.h5" % (stock, genetic_msp, i)
        while model_save_fname in this_dir:
            i += 1
            model_save_fname = "%s-%4.2f-%d-fourier.h5" % (stock, genetic_msp, i)

        model.save(model_save_fname)
        print("Saved model as %s" % (model_save_fname))


def use_existing_h5(fn_h5, train_start, train_end, test_start, test_end, pop_size=3, n_generations=2):
    pass

def evaluate_loss(stock=None, period=None, interval=""):
    pass

def base_rnn(stock=None, period=None, interval=""):
    # Load the bar feed from the CSV file
    if stock is None or period is None:
        stock = sys.argv[1].lower()
        period = sys.argv[2].lower() if len(sys.argv) >= 3 else "1y"
        interval = sys.argv[3].lower() if len(sys.argv) >= 4 else ""

    # TODO segment into testing and training feeds
    path = ""
    if len(interval) == 0:
        path = "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower())
    else:
        path = "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), period.lower(), interval.lower())

    X_train, y_train, norm_y_train = csv_to_xy(path)

    feed = quandlfeed.Feed() if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
    feed.addBarsFromCSV(stock, path)

    model = Sequential()
    if use_ind_values:
        test_strat = RNNStrategy(feed, stock, None, is_test=True)
        test_strat.run()
        ind_values = test_strat.get_ind_values()[1:]

        X_train = [ind_values[i-window_size:i] for i in range(window_size, len(ind_values))]
        X_train = np.reshape(np.array(X_train), (len(X_train), window_size, n_models, 1))

        # TODO turn this into RNN with n_models inputs per time step, using window_size
        # TODO use separate model to predict max_spend (volatility)

        model.add(LSTM(lstm_units, input_shape=(window_size, n_models,), activation='linear'))
        #model.add(SimpleRNN(hidden_units, input_shape=(window_size, n_models,), activation='linear'))
        #model.add(InputLayer(input_shape=(window_size, n_models,)))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        #model.summary()

        y_train = y_train[window_size:]
        norm_y_train = norm_y_train[window_size:]

        #print(X_train, norm_y_train)

        # TODO go back to norm_y_train ?
        model.fit(X_train, norm_y_train, epochs=n_epochs)

    else:
        # TODO train RNN outside strategy framework, then build strategy for testing side
        # TODO figure out how to use indicator values in training later on

        model.add(LSTM(lstm_units, input_shape=(window_size, 1), activation='linear'))
        #model.add(SimpleRNN(hidden_units, input_shape=(window_size, 1), activation='linear'))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=dense_units, activation='linear'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()

        norm_y_train = norm_y_train[window_size:]

        model.fit(X_train, norm_y_train, epochs=n_epochs)

    # TODO create command line framework for train/test demarcation

    feed = quandlfeed.Feed() if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
    feed.addBarsFromCSV(stock, path)

    strat = RNNStrategy(feed, stock, model, is_test=False, verbose=True)
    strat.run()

    final_port_value = strat.getBroker().getEquity()
    print("Final portfolio value: $%.2f" % final_port_value)

    plt.title(stock.upper()+" using RNN Strategy")
    plt.plot([i for i in range(len(strat.get_port_values()))], strat.get_port_values(), label="Port Value")
    plt.plot([i for i in range(len(strat.get_port_values()))], [p*int(total_cash/strat.get_cprices()[0]) for p in strat.get_cprices()], label="Adj Share Price")
    plt.plot([i for i in range(len(strat.get_port_values()))], strat.get_share_values(), label="Port Value in Shares")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if sys.argv[1][-3:] == '.h5':
        use_existing_h5(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        #genetic_train_and_test_rnn(sys.argv[1].lower(), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

        fourier_series_rnn(sys.argv[1].lower(), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        #csv_to_xy_fourier("WIKI-ZG-1y-yfinance.csv")