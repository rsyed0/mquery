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

import utilities
from yfinance_csv import fetch_price_csv

n_bins = 10
bin_width = 1 / n_bins

window_size = 10
batch_size = 5
learning_rate = 0.1
n_epochs = 25
total_cash = 100000
interval_pct_max = 0.25
max_spend = 1 #0.25

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Using fixed context window classifier
# GOAL: predict next normalized price

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


    def forward(self, data):
        batch_size, window_size = data.shape

        assert window_size == self.length

        # using log-domain
        log_leafs = nn.functional.log_softmax(self.input_distros, dim=-1)

        data.apply_(xtb)

        y = log_leafs[:, data[:,0].long()]

        # working w 1d values instead of scalars
        for layer_i in range(self.length-1):
            y = torch.logsumexp(y + nn.functional.log_softmax(self.dense_layer_weights[layer_i], dim=-1), dim=1) + log_leafs[:, data[:,layer_i+1].long()]

        y = torch.exp(torch.logsumexp(y, dim=0))

        return y


def xtb(x):
    return int(min(n_bins-1, x // bin_width))


class HMMStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, model, min_price, max_price):
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
        self.__window.append((c_price - self.__minPrice) / (self.__maxPrice - self.__minPrice))

        if len(self.__window) < window_size:
            return
        else:
            self.__window.pop(0)

        #print(self.__window)
        
        # TODO integrate Pr(X1 ... Xn) over Xn in [0,1]
        # TODO decide how/if to batch-ify testing data (currently submitting as batches of 1)
        components = [self.__model(torch.Tensor([self.__window + [x_n * bin_width]])).cpu().detach().numpy()[0] * bin_width for x_n in range(0, n_bins+1)]
        sm_c = sum(components)
        xn_distro = [c_i / sm_c for c_i in components]
        xn_mean = sum([xn_distro[i] * i / n_bins for i in range(n_bins+1)])

        self.__xnDistros.append(xn_distro)

        #print(xn_mean)

        buy_sell = min(1, max(-1, (xn_mean - self.__window[-1]) / interval_pct_max))
        delta_shares = int(max_spend * total_cash * buy_sell)

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

def main():
    # PT 1: read in and normalize price values
    stock_path, stock, period, interval = utilities.get_stock_path(sys.argv)

    model = None
    mn_sp, mx_sp = -1, -1
    if sys.argv[-1][-3:] == ".pt":

        # TODO test this
        model = get_existing_model(sys.argv[-1])
        _, mn_sp_str, mx_sp_str, _, _ = sys.argv[-1].split("-")
        mn_sp, mx_sp = float(mn_sp_str), float(mx_sp_str)
    else:
        try:
            stock_df = pd.read_csv(stock_path)
        except FileNotFoundError:
            if len(interval) == 0:
                fetch_price_csv(stock.upper(), period.lower())
            else:
                fetch_price_csv(stock.upper(), period.lower(), interval.lower())
            stock_df = pd.read_csv(stock_path)

        stock_prices = stock_df['Close']
        mn_sp, mx_sp = min(stock_prices), max(stock_prices)
        norm_stock_prices = [(x-mn_sp) / (mx_sp-mn_sp) for x in stock_prices]

        # PT 2: convert to pytorch dataset/tensors
        windows = [norm_stock_prices[i:i+window_size] for i in range(len(norm_stock_prices) - window_size)]

        #print(windows)

        # TODO debug this line
        #print([windows[i*batch_size:(i+1)*batch_size] for i in range(len(windows) // batch_size)])
        batches = torch.Tensor([windows[i*batch_size:(i+1)*batch_size] for i in range(len(windows) // batch_size)])

        #print(batches)

        # PT 3: train FCW with optim.step()
        model = HMM(batch_size, window_size)
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

        lls = []

        for epoch_i in range(n_epochs):
            epoch_lls = []

            for batch in batches:
                optim.zero_grad()

                batch = batch.to(device)
                batch_lls = model(batch)

                loss = -torch.sum(batch_lls, dim=-1)
                loss.retain_grad()

                loss.backward()
                optim.step()

                epoch_lls.extend(torch.flatten(batch_lls).tolist())

            lls.append(epoch_lls)

        lls = torch.Tensor(lls)
        #print(lls)
        model.to('cpu')
        ll = torch.mean(lls, dim=-1)
        print(ll)
        print(torch.max(lls))

    # PT 4: create strategy, plot results
    feed = utilities.get_stock_feed(stock, stock_path, interval)

    # TODO decide if can use mn_sp and mx_sp for testing data normalization
    hmm_strategy = HMMStrategy(feed, stock, model, mn_sp, mx_sp)
    hmm_strategy.run()

    plt.title("%s with FCW Analysis" % (stock.upper()))
    plt.plot([i for i in range(len(hmm_strategy.get_port_values()))], hmm_strategy.get_port_values(), label="Port Value")
    plt.plot([i for i in range(len(hmm_strategy.get_port_values()))], [p*int(total_cash/hmm_strategy.get_cprices()[0]) for p in hmm_strategy.get_cprices()], label="Adj Share Price")
    plt.plot([i for i in range(len(hmm_strategy.get_port_values()))], hmm_strategy.get_share_values(), label="Port Value in Shares")
    plt.legend()
    plt.show()

    print(hmm_strategy.get_buy_sell_rtgs())

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