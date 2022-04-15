from __future__ import print_function

from pyalgotrade import strategy, plotter
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.technical import ma, rsi, macd, bollinger

from pyalgotrade.bar import Frequency
from pyalgotrade.barfeed.csvfeed import GenericBarFeed

import matplotlib.pyplot as plt

from math import isclose

from random import randint, random

from copy import deepcopy

from strategies import *

import sys

#pop_size = 30
#n_generations = 10
n_survivors = 10
p_mutation = 0.5
msp_dominance = 0.5  # 0.5 to 1

allow_negation = False
n_models = 7*(2 if allow_negation else 1)

keep_best_model = False

model_descs = ['SMA','RSI','SmaRsi','MACD','CBasis','GFill','History']
if allow_negation:
    nmd = []
    for s in model_descs:
        nmd.append(s)
        nmd.append("-"+s)
    model_descs = nmd

total_cash = 10000

class WeightedIndicatorStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, weights, smaPeriod=15, max_spend=0.25, live=False, fastEmaPeriod=12, slowEmaPeriod=26, signalEmaPeriod=9, bBandsPeriod=1, verbose=False):
        super(WeightedIndicatorStrategy, self).__init__(feed, 10000)
        self.__position = None
        self.__instrument = instrument

        self.__weights = weights
        self.__onBars = [sma_onBars, rsi_onBars, smarsi_onBars, macd_onBars, cbasis_onBars, gfill_onBars, history_onBars]
        self.__cbasis = 0

        self.__modelWtValues = None

        assert isclose(sum(self.__weights), 1)
        if allow_negation:
            assert len(self.__weights) == len(self.__onBars)*2
        else:
            assert len(self.__weights) == len(self.__onBars)

        self.__lastTop, self.__lastBottom, self.__trend, self.__trendLength = None, None, 0, 0
        self.__portValues, self.__cprices, self.__shareValues = [], [], []
        self.__maxSpend = max_spend

        self.__sma = ma.SMA(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__rsi = rsi.RSI(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__macd = macd.MACD(feed[instrument].getPriceDataSeries(), fastEmaPeriod, slowEmaPeriod, signalEmaPeriod)
        self.__bb = bollinger.BollingerBands(feed[instrument].getPriceDataSeries(), bBandsPeriod, 2)

        self.__verbose = verbose

        # TODO decide if this is needed
        if live:
            # Subscribe to order book update events to get bid/ask prices to trade.
            feed.getOrderBookUpdateEvent().subscribe(self.__onOrderBookUpdate)

    def __onOrderBookUpdate(self, orderBookUpdate):
        bid = orderBookUpdate.getBidPrices()[0]
        ask = orderBookUpdate.getAskPrices()[0]

        if bid != self.__bid or ask != self.__ask:
            self.__bid = bid
            self.__ask = ask
            self.info("Order book updated. Best bid: %s. Best ask: %s" % (self.__bid, self.__ask))

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

    def get_model_wt_values(self):
        return self.__modelWtValues

    def set_last_top(self, lt):
        self.__lastTop = lt

    def set_last_bottom(self, lb):
        self.__lastBottom = lb

    def set_trend(self, t):
        self.__trend = t

    def set_trend_length(self, tl):
        self.__trendLength = tl

    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        self.info("BUY at $%.2f" % (execInfo.getPrice()))

    def onEnterCanceled(self, position):
        self.__position = None

    def onExitOk(self, position):
        execInfo = position.getExitOrder().getExecutionInfo()
        self.info("SELL at $%.2f" % (execInfo.getPrice()))
        self.__position = None

    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        self.__position.exitMarket()

    # IMPLEMENTATION OF STRATEGY
    # TODO save values of each indicator along data interval to avoid recalculation
    def onBars(self, bars):
        n_shares = self.getBroker().getShares(self.get_instrument())
        bar = bars[self.get_instrument()]
        c_price = bar.getPrice()
        strat_cash = self.getBroker().getCash(False)

        if not allow_negation:
            # w/o inversing indicators
            model_wt_values = [wt*st_onBars(self, bars) for wt,st_onBars in zip(self.__weights, self.__onBars)]
            delta_shares = int(total_cash*self.__maxSpend*sum(model_wt_values)/c_price)
            self.__modelWtValues = model_wt_values
        else:
            # w/ inversing indicators
            model_wt_values = []
            wt_sum = 0
            for i in range(len(self.__onBars)):
                res = self.__onBars[i](self, bars)
                wt_sum += (self.__weights[2*i]*res - self.__weights[2*i+1]*res)
                model_wt_values.extend([self.__weights[2*i]*res, -self.__weights[2*i+1]*res])
            delta_shares = int(total_cash*self.__maxSpend*wt_sum/c_price)
            self.__modelWtValues = model_wt_values

        if delta_shares > 0 and strat_cash < delta_shares*c_price:
            delta_shares = int(strat_cash/c_price)

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
        self.__shareValues.append(n_shares*c_price)

def cross(model_a, model_b, n_children=2):
    # equal weighted sum of both probs
    model_c = [(a+b)/2 for a,b in zip(model_a[0], model_b[0])]

    max_spend = (msp_dominance*model_a[1]+(1-msp_dominance)*model_b[1])

    # plus random mutation(s)
    out = []
    for nc in range(n_children):
        mc = deepcopy(model_c)
        for i in range(n_models):
            if random() < p_mutation:

                # experiment with different mutation systems
                i,j = -1,-1
                while i == j:
                    i,j = randint(0,n_models-1), randint(0,n_models-1)

                delta = random()*mc[j]
                mc[j] -= delta
                mc[i] += delta

                #mc[i] += random()*size_mutation

        out.append((mc, max_spend))
    return out

def normalize(model):
    # make so that all vals sum to 1
    s = sum(model)
    return [x/s for x in model]

def run_strategy(stock=None, period=None, interval=None, pop_size=30, n_generations=10, verbose=True):
    # Load the bar feed from the CSV file
    #feed = quandlfeed.Feed()
    if stock is None or period is None or interval is None:
        stock = sys.argv[1].lower()
        period = sys.argv[2].lower() if len(sys.argv) >= 3 else "1y"
        interval = sys.argv[3].lower() if len(sys.argv) >= 4 else ""

    #feed.addBarsFromCSV(stock, "WIKI-"+stock.upper()+"-1y-yfinance.csv")

    # Evaluate the strategy with the feed.
    # eg. sma+rsi best for GME 1y (60% return), cbasis best for AAPL 1y (16% return)

    # use genetic algorithm to fine tune weights parameter for a given stock

    # TODO try tuning max_spend by bin search
    # TODO try tuning max_spend by closeness to max profit, trading volume, etc

    # structure: model = ([weights], max_spend)
    population = []
    for i in range(pop_size):
        probs = [0.0 for j in range(n_models)]
        n, pl = n_models, 1.0
        while n > 0 and pl > 0.0:
            ri = randint(0,n_models-1)
            if probs[ri] == 0.0:
                probs[ri] = random()*pl
                pl -= probs[ri]
                n -= 1
        population.append((normalize(probs),random()))

    #print(population)
    high_score = 0
    best_model = None

    for i in range(n_generations):

        scores = [0 for i in range(pop_size)]
        for p_i in range(pop_size):

            feed = quandlfeed.Feed() if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
            if len(interval) == 0:
                feed.addBarsFromCSV(stock, "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower()))
            else:
                feed.addBarsFromCSV(stock, "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), period.lower(), interval.lower()))

            # run strategy, save score
            strat = WeightedIndicatorStrategy(feed, stock, population[p_i][0], max_spend=population[p_i][1]) #verbose=p_i==0)
            strat.run()
            score = strat.getBroker().getEquity()
            #print(population[p_i],score)
            scores[p_i] = score

        scores = [(i,scores[i]) for i in range(pop_size)]
        scores.sort(key=lambda x: x[1], reverse=True)

        if verbose:
            print("Generation",i+1,":",scores)

        scores = scores[:n_survivors]
        if scores[0][1] > high_score:
            best_model = population[scores[0][0]]
            high_score = scores[0][1]

        next_pop = [best_model] if keep_best_model else []
        for i in range(len(scores)):
            for j in range(i+1, len(scores)):
                res = cross(population[scores[i][0]], population[scores[j][0]], n_children=1)
                next_pop.extend([(normalize(x[0]),x[1]) for x in res])
            if len(next_pop) >= pop_size:
                population = next_pop[:pop_size]
                break

    feed = quandlfeed.Feed() if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
    if len(interval) == 0:
        feed.addBarsFromCSV(stock, "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower()))
    else:
        feed.addBarsFromCSV(stock, "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), period.lower(), interval.lower()))

    myStrategy = WeightedIndicatorStrategy(feed, stock, best_model[0], max_spend=best_model[1], verbose=verbose)
    myStrategy.run()

    final_port_value = myStrategy.getBroker().getEquity()
    if verbose:
        print("Final portfolio value: $%.2f" % final_port_value)

    model_scores = sorted([(model_descs[i],best_model[0][i]) for i in range(n_models)], key=lambda x:x[1], reverse=True)
    best_model_desc = str([(x,str(y)[:4]) for x,y in model_scores[:4]])

    if verbose:
        print(best_model)
        print(best_model_desc)

    if allow_negation:
        print("Positive:",sum([best_model[0][i] for i in range(0,len(best_model[0]),2)]),", Negative:",sum([best_model[0][i] for i in range(1,len(best_model[0]),2)]))

    if verbose:
        plt.title(stock.upper()+" "+best_model_desc) #str([str(x)[:4] for x in best_model]))
        plt.plot([i for i in range(len(myStrategy.get_port_values()))], myStrategy.get_port_values(), label="Port Value")
        #plt.show()
        plt.plot([i for i in range(len(myStrategy.get_port_values()))], [p*int(total_cash/myStrategy.get_cprices()[0]) for p in myStrategy.get_cprices()], label="Adj Share Price")
        plt.plot([i for i in range(len(myStrategy.get_port_values()))], myStrategy.get_share_values(), label="Port Value in Shares")
        plt.legend()
        plt.show()

    return myStrategy, best_model[1]

if __name__ == "__main__":
    run_strategy()