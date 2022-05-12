from pyalgotrade import strategy, plotter
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.technical import ma, rsi, macd, bollinger

from pyalgotrade.bar import Frequency
from pyalgotrade.barfeed.csvfeed import GenericBarFeed

#from pat_papertrade import WeightedIndicatorStrategy

import matplotlib.pyplot as plt

from math import isclose

from random import randint, random

from copy import deepcopy

from yfinance_csv import fetch_price_csv

from strategies import *

import sys

pop_size = 30
n_generations = 10
n_survivors = 10
p_mutation = 0.5
msp_dominance = 0.5  # 0.5 to 1

allow_negation = True
n_models = 7*(2 if allow_negation else 1)

keep_best_model = False

total_cash = 100000

fastEmaPeriod = 12
slowEmaPeriod = 26
signalEmaPeriod = 9
bBandsPeriod = 1
smaPeriod = 15

default_max_spend = 0.25

# TODO change to centralized model
class MultipleWeightedIndicatorStrategy(strategy.BacktestingStrategy):
    def __init__(self, feeds, ovr_feed, instruments, weights_2d, max_spend=default_max_spend, total_cash=total_cash, verbose=False):
        super(MultipleWeightedIndicatorStrategy, self).__init__(ovr_feed, total_cash)
        
        # needed ?
        #self.__strategies = {instrument: WeightedIndicatorStrategy(feed, instrument, weights, total_cash=total_cash, trade=False, parent_strat=self) for feed, instrument, weights in zip(feeds, instruments, weights_2d)}
        #self.__cash = total_cash
        self.__modelWtValues = {}
        self.__feeds = feeds

        self.__instruments = instruments
        self.__weights = {instrument: weights for instrument, weights in zip(instruments, weights_2d)}
        self.__onBars = [sma_onBars, rsi_onBars, smarsi_onBars, macd_onBars, cbasis_onBars, gfill_onBars, history_onBars]
        self.__cbasis = {instrument: 0 for instrument in instruments}

        self.__cprices = {instrument: [] for instrument in instruments}
        self.__nshares = {instrument: [] for instrument in instruments}

        for weights in weights_2d:
            assert isclose(sum(weights), 1)
            if allow_negation:
                assert len(weights) == len(self.__onBars)*2
            else:
                assert len(weights) == len(self.__onBars)

        self.__sma = {instrument: ma.SMA(ovr_feed[instrument].getPriceDataSeries(), smaPeriod) for instrument in instruments}
        self.__rsi = {instrument: rsi.RSI(ovr_feed[instrument].getPriceDataSeries(), smaPeriod) for instrument in instruments}
        self.__macd = {instrument: macd.MACD(ovr_feed[instrument].getPriceDataSeries(), fastEmaPeriod, slowEmaPeriod, signalEmaPeriod) for instrument in instruments}
        self.__bb = {instrument: bollinger.BollingerBands(ovr_feed[instrument].getPriceDataSeries(), bBandsPeriod, 2) for instrument in instruments}

        # TODO implement per-instrument max_spend
        self.__maxSpend = max_spend

        # TODO track share values/n_shares for each instrument
        self.__shareValues, self.__cashValues, self.__portValues = [], [], []

        self.__verbose = verbose

    # "override" for decentralized approach
    """def run_strategy(self):
        # TODO decide how to use ovr_feed
        for stock, strat in self.__strategies.items():
            print(stock)
            strat.run()"""

    def get_sma(self, instrument):
        return self.__sma[instrument]

    def get_rsi(self, instrument):
        return self.__rsi[instrument]

    def get_macd(self, instrument):
        return self.__macd[instrument]

    def get_bb(self, instrument):
        return self.__bb[instrument]

    def get_cbasis(self, instrument):
        return self.__cbasis[instrument]

    def get_cprices(self, instrument):
        return self.__cprices[instrument]

    def get_nshares(self, instrument):
        return self.__nshares[instrument]

    def get_cash_value(self):
        #share_value = self.get_share_value()
        #port_value = self.get_port_value()
        #return port_value - share_value

        return self.getBroker().getCash()

    def get_cash_values(self):
        return self.__cashValues

    def get_share_value(self):
        for instrument in self.__instruments:
            if len(self.__cprices[instrument]) == 0:
                return -1

        return sum([self.__cprices[instrument][-1]*self.getBroker().getShares(instrument) for instrument in self.__weights.keys()])

    def get_share_values(self):
        return self.__shareValues

    def get_port_value(self):
        return self.getBroker().getEquity()

    def get_port_values(self):
        return self.__portValues
    
    """def notify(self, instrument, time, model_wt_values):
        if time in self.__modelWtValues:
            self.__modelWtValues[time][instrument] = sum(model_wt_values)

            if len(self.__modelWtValues[time]) == len(self.__strategies) and len(self.__portValues) == time:
                buys = []
                for stock, buy_sell in self.__modelWtValues[time].items():
                    n_shares = self.__strategies[stock].get_n_shares()

                    # sells then buys
                    if buy_sell < 0:
                        delta_shares = buy_sell * total_cash
                        self.__cash += self.__strategies[stock].trade_shares(delta_shares)
                    else:
                        positives.append((stock,buy_sell))

                # normalize all buy signals to total cash on hand * max_spend parameter
                s = sum([x for _,x in buys])
                positives = [(k,v/s) for k,v in buys]
                spend_cash = self.__maxSpend * self.__cash

                for stock, norm_buy_sell in buys.items():
                    self.__strategies[stock].trade_cash(spend_cash * norm_buy_sell)

                self.__shareValues.append(self.get_share_value())
                self.__cashValues.append(self.get_cash_value())
                self.__portValues.append(self.get_port_value())
        else:
            self.__modelWtValues[time] = {instrument: sum(model_wt_values)}"""

    def onBars(self, bars):
        c_prices = {instrument: bar.getPrice() for instrument, bar in bars.items()}

        buy_sell_rtgs = {}
        for instrument in self.__instruments:
            weights = self.__weights[instrument]
            model_wt_values = []

            if allow_negation:
                for i in range(len(self.__onBars)):
                    res = self.__onBars[i](self, bars, instrument)
                    model_wt_values.extend([weights[2*i]*res, -weights[2*i+1]*res])
            else:
                model_wt_values = [wt*st_onbars(self, bars, instrument) for wt, st_onBars in zip(weights, self.__onBars)]

            buy_sell_rtgs[instrument] = sum(model_wt_values)

        # get total value of all shares
        share_value = self.get_share_value()

        buys = []
        for instrument, buy_sell in buy_sell_rtgs.items():
            n_shares = self.getBroker().getShares(instrument)
            instrument_share_value = n_shares * c_prices[instrument]
            exposure = instrument_share_value / share_value if not share_value == 0 else 0

            # sells then buys
            # TODO consider swapping order, sell orders should be processed pre-buy to add to cash on hand
            if buy_sell < 0:
                # TODO find a better metric (cash -> shares)
                delta_shares = max(-n_shares, int(buy_sell * (exposure * total_cash) / c_prices[instrument]))

                if not delta_shares == 0:
                    if self.__verbose:
                        print("Day %d: sold %d shares of %s at $%-7.2f" % (len(self.__cprices[instrument])+1, abs(delta_shares), instrument.upper(), c_prices[instrument]))
                    self.marketOrder(instrument, delta_shares)
                    self.__cbasis[instrument] = (n_shares*self.__cbasis[instrument] + delta_shares*c_prices[instrument]) / (n_shares + delta_shares) if not n_shares + delta_shares == 0 else 0

            elif buy_sell > 0:
                buys.append((instrument, buy_sell))

        # normalize all buy signals to total cash on hand * max_spend parameter
        s = sum([x for _,x in buys])
        buys = [(k,v/s) for k,v in buys]
        spend_cash = self.__maxSpend * self.get_cash_value()

        #print(buys)
        if spend_cash < 0:
            print("FATAL ERROR: spend_cash < 0")
            sys.exit(1)

        for instrument, norm_buy_sell in buys:
            #self.__strategies[stock].trade_cash(spend_cash * norm_buy_sell)
            delta_shares = int(spend_cash * norm_buy_sell / c_prices[instrument])

            if not delta_shares == 0:
                if self.__verbose:
                    print("Day %d: bought %d shares of %s at $%7.2f" % (len(self.__cprices[instrument])+1, delta_shares, instrument.upper(), c_prices[instrument]))
                self.marketOrder(instrument, delta_shares)
                self.__cbasis[instrument] = (n_shares*self.__cbasis[instrument] + delta_shares*c_prices[instrument]) / (n_shares + delta_shares)

        for instrument, c_price in c_prices.items():
            self.__cprices[instrument].append(c_price)
            self.__nshares[instrument].append(self.getBroker().getShares(instrument))

        self.__shareValues.append(self.get_share_value())
        self.__cashValues.append(self.get_cash_value())
        self.__portValues.append(self.get_port_value())

def normalize(model):
    # make so that all vals sum to 1
    s = sum(model)
    return [x/s for x in model]

def cross(model_a, model_b, n_children=2):
    # equal weighted sum of both probs
    model_c = [[(a+b)/2 for a,b in zip(ar,br)] for ar,br in zip(model_a[0],model_b[0])]

    max_spend = (msp_dominance*model_a[1]+(1-msp_dominance)*model_b[1])

    # plus random mutation(s)
    out = []
    for nc in range(n_children):
        mc = deepcopy(model_c)
        for r in range(len(mc)):
            for i in range(n_models):
                if random() < p_mutation:

                    # experiment with different mutation systems
                    i,j = -1,-1
                    while i == j:
                        i,j = randint(0,n_models-1), randint(0,n_models-1)

                    delta = random()*mc[r][j]
                    mc[r][j] -= delta
                    mc[r][i] += delta

                    #mc[i] += random()*size_mutation

        out.append((mc, max_spend))
    return out

def main():
    if len(sys.argv) < 2:
        sys.exit()

    args = sys.argv[1:]
    symbols = []
    this_symbol = []
    for arg in args:
        if arg[0].isnumeric():
            if len(this_symbol) == 1:
                symbols.append((this_symbol[0], arg))
                this_symbol = []
            else:
                print("invalid args")
                sys.exit(1)
        else:
            if len(this_symbol) == 0:
                this_symbol.append(arg.lower())
            elif len(this_symbol) == 1:
                symbols.append((this_symbol[0], '1y'))
                this_symbol = [arg.lower()]

    if len(this_symbol) == 1:
        symbols.append((this_symbol[0], '1y'))
    elif len(this_symbol) == 2:
        symbols.append(this_symbol)

    print(symbols)

    # TODO debug issues when different periods are used
    orig_feeds = []
    orig_ovr_feed = quandlfeed.Feed()
    for stock, period in symbols:
        feed = quandlfeed.Feed()
        feed.addBarsFromCSV(stock, "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower()))
        orig_ovr_feed.addBarsFromCSV(stock, "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower()))
        orig_feeds.append(feed)

    instruments = [s.lower() for s,_ in symbols]
    n_symbols = len(symbols)

    population = []
    for i in range(pop_size):
        pop_member = []
        for j in range(n_symbols):
            pop_member.append(normalize([random() for i in range(n_models)]))
        population.append((pop_member,random()))

    high_score, best_model = 0, None        

    for i in range(n_generations):

        scores = []
        for j in range(pop_size):
            feeds = deepcopy(orig_feeds)
            ovr_feed = deepcopy(orig_ovr_feed)
            this_strategy = MultipleWeightedIndicatorStrategy(feeds, ovr_feed, instruments, population[j][0], max_spend=population[j][1])
            this_strategy.run()

            score = this_strategy.get_port_value()
            scores.append((j, score))

        scores.sort(key=lambda x:x[1], reverse=True)

        print("Generation",i+1,":",scores)

        scores = scores[:n_survivors]
        if scores[0][1] > high_score:
            best_model = population[scores[0][0]]
            high_score = scores[0][1]

        next_pop = [best_model] if keep_best_model else []
        for i in range(len(scores)):
            for j in range(i+1, len(scores)):
                a,b = population[scores[i][0]], population[scores[j][0]]

                #print(a,b)
                #res = ([cross(ar, br, n_children=1) for ar,br in zip(a[0],b[0])], msp_dominance*a[1] + (1-msp_dominance)*b[1])
                #next_pop.append(([normalize(x) for x in res[0]], res[1]))

                res = cross(a, b, n_children=1)
                next_pop.extend(res)
            if len(next_pop) >= pop_size:
                population = next_pop[:pop_size]
                break

    feeds = deepcopy(orig_feeds)
    ovr_feed = deepcopy(orig_ovr_feed)
    best_strategy = MultipleWeightedIndicatorStrategy(feeds, ovr_feed, instruments, best_model[0], max_spend=best_model[1])
    best_strategy.run()

    print(best_model)
    print("Final portfolio value: $%-7.2f" % (best_strategy.get_port_value()))

    t_axis = [i for i in range(len(best_strategy.get_port_values()))]

    plt.title([x[0] for x in symbols]) #str([str(x)[:4] for x in best_model]))
    plt.plot(t_axis, best_strategy.get_port_values(), label="Port Value")
    plt.plot(t_axis, best_strategy.get_cash_values(), label="Cash Position")
    plt.plot(t_axis, best_strategy.get_share_values(), label="Value in Shares")

    for instrument in instruments:
        plt.plot(t_axis, [n_shares*c_price for n_shares, c_price in zip(best_strategy.get_nshares(instrument), best_strategy.get_cprices(instrument))], label=instrument.upper()+" Port Value")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()