import torch
from torch import nn

from pyalgotrade import strategy, plotter
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.technical import ma, rsi, macd, bollinger

from pyalgotrade.bar import Frequency
from pyalgotrade.barfeed.csvfeed import GenericBarFeed

from strategies import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os

from copy import deepcopy

from math import exp, isclose
from statistics import mean, stdev

import utilities
from yfinance_csv import fetch_price_csv

# TODO add interval-based multiplier to max_spend

n_bins = 50
bin_width = 1 / n_bins

window_size = 10
batch_size = 5
learning_rate = 0.1
n_epochs = 25
total_cash = 100000
interval_pct_max = 0.25
max_spend = 1 #0.25
delta_pct_max = 0.05

# TODO gives similar output for all inputs when using deltas
# might just be non-predictable using day-to-day deltas, which are volatile
# may need to average out over intervals
# or need larger input dataset size
use_deltas = True
use_normal = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Using fixed context window classifier
# GOAL: predict next normalized price
# TODO find a way to integrate strategies.py

# TODO use pct chgs between sessions instead of absolute/norm values

# TODO figure out how to turn Pr(X1, ..., Xn) into Pr(Xn | X1, ..., Xn-1)
class HMM(torch.nn.Module):
    def __init__(self, width, length, pr_i=None):
        super().__init__()

        self.width, self.length = width, length
        assert self.width >= 1 and self.length >= 1

        # dimensions: [width, n_bins]
        if pr_i is None:
            pr_i = torch.randn(width, n_bins)
        self.input_distros = nn.Parameter(pr_i, requires_grad=True)

        # dimensions: [length-1, width, width]
        dense_layer_weights = torch.randn(length-1, width, width)
        self.dense_layer_weights = nn.Parameter(dense_layer_weights, requires_grad=True)

        # dimensions: [width]
        final_layer_weights = torch.randn(width)
        self.final_layer_weights = nn.Parameter(final_layer_weights, requires_grad=True)


    def forward(self, data):
        batch_size, window_size = data.shape
        assert window_size == self.length

        # using log-domain
        log_leafs = nn.functional.log_softmax(self.input_distros, dim=-1)

        #print(data)

        # convert normalized data into bin indices
        data.apply_(xtb)

        # get output of first product layer (first set of leaf nodes * [1,1,1,...])
        y = torch.transpose(log_leafs[:, data[:,0].long()], 0, 1)

        # working w 1d values instead of scalars
        for layer_i in range(self.length-1):
            
            # shape: [width, width]
            log_dense_weights = nn.functional.log_softmax(self.dense_layer_weights[layer_i], dim=-1)

            # use broadcasting to compute dense layer weighted sums
            # compute output of sum layer
            sm_y = torch.reshape(y, (batch_size, 1, self.width)) + log_dense_weights
            y = torch.logsumexp(sm_y, dim=-1)

            # get output of leaf/distribution nodes
            # compute output of product layer
            y = y + torch.transpose(log_leafs[:, data[:,layer_i+1].long()], 0, 1)

        log_dense_weights = nn.functional.log_softmax(self.final_layer_weights, dim=0)
        sm_y = y + log_dense_weights
        y = torch.logsumexp(sm_y, dim=1)

        #sys.exit()

        return y


def xtb(x):
    return int(min(n_bins-1, max(0, x // bin_width)))


class HMMStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, model, min_price, max_price, mean_price, std_price):
        super(HMMStrategy, self).__init__(feed, total_cash)
        self.__position = None
        self.__instrument = instrument
        self.__model = model

        # TODO decide if deque is better (O(1) deletion from front)
        self.__window = []
        self.__portValues, self.__cprices, self.__shareValues = [], [], []
        self.__buySellRtgs = []

        self.__xnDistros = []
        self.__minPrice, self.__maxPrice = min_price, max_price
        self.__meanPrice, self.__stdPrice = mean_price, std_price

    def get_xn_distros(self):
        return self.__xnDistros

    def get_port_values(self):
        return self.__portValues

    def get_cprices(self):
        return self.__cprices

    def get_share_values(self):
        return self.__shareValues

    def get_buy_sell_rtgs(self):
        return self.__buySellRtgs

    def onBars(self, bars):
        c_price = bars[self.__instrument].getPrice()
        n_shares = self.getBroker().getShares(self.__instrument)
        strat_cash = self.getBroker().getCash(False)

        # TODO decide how to normalize input data (unbounded)
        window_cprice = c_price
        if use_deltas:
            # TODO debug, always giving similar values...
            if len(self.get_cprices()) == 0:
                window_cprice = 0
            else:
                last_price = self.get_cprices()[-1]
                window_cprice = (c_price - last_price) / last_price

        if use_normal:
            norm_window_cprice = (window_cprice - self.__meanPrice) / self.__stdPrice
            norm_window_cprice = min(1, max(-1, (norm_window_cprice / 2 + 0.5)))
            self.__window.append(norm_window_cprice)
        else:
            self.__window.append((window_cprice - self.__minPrice) / (self.__maxPrice - self.__minPrice))

        if len(self.__window) < window_size:
            return
        else:
            self.__window.pop(0)

        #print(self.__window)
        
        # TODO integrate Pr(X1 ... Xn) over Xn in [0,1]
        # TODO decide how/if to batch-ify testing data (currently submitting as batches of 1)
        components = [exp(self.__model(torch.Tensor([self.__window + [(x_n * bin_width) + (bin_width / 2)]])).cpu().detach().numpy()[0]) for x_n in range(0, n_bins+1)]
        sm_c = sum(components)
        xn_distro = [c_i / sm_c for c_i in components]
        xn_mean = sum([xn_distro[i] * i * bin_width for i in range(n_bins+1)])
        print(xn_mean)

        self.__xnDistros.append(xn_distro)

        buy_sell, delta_shares = None, None
        if use_deltas:
            denorm_xn_mean = (xn_mean - 0.5) * 2 if use_normal else self.__minPrice + xn_mean * (self.__maxPrice - self.__minPrice)
            buy_sell = min(1, max(-1, denorm_xn_mean / delta_pct_max))
            delta_shares = int(max_spend * total_cash * buy_sell / c_price)
        else:
            buy_sell = min(1, max(-1, (xn_mean - self.__window[-1]) / interval_pct_max))
            delta_shares = int(max_spend * total_cash * buy_sell / c_price)

        if delta_shares < 0:
            if n_shares + delta_shares < 0:
                delta_shares = -n_shares
        elif delta_shares > 0:
            if delta_shares * c_price > strat_cash:
                delta_shares = strat_cash // c_price

        print("Day %d: have %d shares and $%-7.2f" % (len(self.__xnDistros), n_shares, strat_cash), end="")
        if delta_shares > 0:
            print(", buying %d shares at $%-7.2f" % (delta_shares, c_price))
        elif delta_shares < 0:
            print(", selling %d shares at $%-7.2f" % (abs(delta_shares), c_price))
        else:
            print("")

        self.marketOrder(self.__instrument, delta_shares)

        self.__portValues.append(self.getBroker().getEquity())
        self.__cprices.append(c_price)
        self.__shareValues.append(n_shares*c_price)
        self.__buySellRtgs.append(buy_sell)

def get_existing_model(model_path):
    model = HMM(batch_size, window_size)
    model.load_state_dict(torch.load(model_path))
    return model

# output shape: (len(prices)-1)
def prices_to_pct_chg(prices):
    return np.array([(prices[i]-prices[i-1]) / prices[i-1] for i in range(1,len(prices))])

def main():
    # PT 1: read in and normalize price values
    stock_path, stock, period, interval = utilities.get_stock_path(sys.argv)

    model = None
    mn_sp, mx_sp = -1, -1
    if sys.argv[-1][-3:] == ".pt":

        # TODO test loading model from local .pt file
        model = get_existing_model(sys.argv[-1])
        _, mn_sp_str, mx_sp_str, _, _ = sys.argv[-1].split("-")
        mn_sp, mx_sp = float(mn_sp_str), float(mx_sp_str)

        # TODO run model
    else:
        try:
            stock_df = pd.read_csv(stock_path)
        except FileNotFoundError:
            if len(interval) == 0:
                fetch_price_csv(stock.upper(), period.lower())
            else:
                fetch_price_csv(stock.upper(), period.lower(), interval.lower())
            stock_df = pd.read_csv(stock_path)

        stock_prices = None
        if use_deltas:
            stock_prices = prices_to_pct_chg(stock_df['Close'])
        else:
            stock_prices = stock_df['Close']

        print(stock_prices)
        mn_sp, mx_sp = min(stock_prices), max(stock_prices)
        av_sp, sd_sp = mean(stock_prices), stdev(stock_prices)
        norm_stock_prices = [min(1, max(-1, ((x-av_sp) / sd_sp) / 2 + 0.5)) for x in stock_prices] if use_normal else [(x-mn_sp) / (mx_sp-mn_sp) for x in stock_prices]
        print(norm_stock_prices)

        #print(stock_prices, norm_stock_prices)

        # PT 2: convert to pytorch dataset/tensors
        windows = [norm_stock_prices[i:i+window_size] for i in range(len(norm_stock_prices) - window_size)]

        #print(windows)

        # TODO debug this line
        #print([windows[i*batch_size:(i+1)*batch_size] for i in range(len(windows) // batch_size)])
        batches = torch.Tensor([windows[i*batch_size:(i+1)*batch_size] for i in range(len(windows) // batch_size)])

        # PT 3: train FCW with optim.step()
        model = HMM(batch_size, window_size)
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

        lls = []

        for epoch_i in range(n_epochs):
            epoch_lls = []

            batches_cp = deepcopy(batches)
            for batch in batches_cp:
                optim.zero_grad()

                batch = batch.to(device)
                batch_lls = model(batch)

                loss = -torch.sum(batch_lls, dim=-1)
                loss.retain_grad()

                loss.backward()
                optim.step()

                epoch_lls.extend(torch.flatten(batch_lls).tolist())

            lls.append(epoch_lls)

            #sys.exit()

        lls = torch.Tensor(lls)
        #print(lls)
        model.to('cpu')
        ll = torch.mean(lls, dim=-1)
        print(ll)
        print(torch.max(lls))

    # PT 4: create strategy, plot results
    feed = utilities.get_stock_feed(stock, stock_path, interval)

    # TODO decide if can use mn_sp and mx_sp for testing data normalization
    hmm_strategy = HMMStrategy(feed, stock, model, mn_sp, mx_sp, av_sp, sd_sp)
    hmm_strategy.run()

    plt.title("%s with FCW Analysis" % (stock.upper()))
    plt.plot([i for i in range(len(hmm_strategy.get_port_values()))], hmm_strategy.get_port_values(), label="Port Value")
    plt.plot([i for i in range(len(hmm_strategy.get_port_values()))], [p*int(total_cash/hmm_strategy.get_cprices()[0]) for p in hmm_strategy.get_cprices()], label="Adj Share Price")
    plt.plot([i for i in range(len(hmm_strategy.get_port_values()))], hmm_strategy.get_share_values(), label="Port Value in Shares")
    plt.legend()
    plt.show()

    buy_sell_rtgs = hmm_strategy.get_buy_sell_rtgs()
    #print(mean(buy_sell_rtgs), stdev(buy_sell_rtgs))
    print(buy_sell_rtgs)

    #print(hmm_strategy.get_xn_distros())
    final_port_value = hmm_strategy.getBroker().getEquity()
    print("Final portfolio value: $%.2f" % final_port_value)

    to_save = input("Save model? (Y or [N]): ")
    if to_save.upper() == "Y":
        this_dir = set(os.listdir())

        i = 1
        model_save_fname = "%s-%.2f-%.2f-%4.2f-%d.pt" % (stock.upper(), mn_sp, mx_sp, max_spend, i)
        while model_save_fname in this_dir:
            i += 1
            model_save_fname = "%s-%.2f-%.2f-%4.2f-%d.pt" % (stock.upper(), mn_sp, mx_sp, max_spend, i)

        torch.save(model.state_dict(), model_save_fname)
        print("Saved model as %s" % (model_save_fname))

if __name__ == "__main__":
    main()