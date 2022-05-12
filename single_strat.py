from __future__ import print_function

from pyalgotrade import strategy, plotter
from pyalgotrade.barfeed import quandlfeed, csvfeed
from pyalgotrade.technical import ma, rsi, macd, bollinger
from pyalgotrade.bar import Frequency

import matplotlib.pyplot as plt

from math import isclose

from random import randint, random

from copy import deepcopy

from strategies import *

import sys

total_cash = 100000
max_spend = 1 #0.25 #0.4377

smaPeriod = 15

class MyStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, smaPeriod, weights, live=False, fastEmaPeriod=12, slowEmaPeriod=26, signalEmaPeriod=9, bBandsPeriod=1, verbose=False):
        super(MyStrategy, self).__init__(feed, total_cash)
        self.__position = None
        self.__instrument = instrument

        # TODO decide whether this is needed
        #self.setUseAdjustedValues(True)

        self.__weights = weights
        self.__onBars = [sma_onBars, rsi_onBars, smarsi_onBars, macd_onBars, cbasis_onBars, gfill_onBars, history_onBars, energy_onBars, dipbuy_onBars]
        self.__cbasis = 0
        assert len(self.__weights) == len(self.__onBars) and isclose(sum(self.__weights), 1)

        self.__lastTop, self.__lastBottom, self.__trend, self.__trendLength = None, None, 0, 0
        self.__portValues, self.__shareValues, self.__cprices = [], [], []

        self.__startPrice, self.__startTime = -1, -1

        self.__sma = ma.SMA(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__rsi = rsi.RSI(feed[instrument].getPriceDataSeries(), smaPeriod)
        self.__macd = macd.MACD(feed[instrument].getPriceDataSeries(), fastEmaPeriod, slowEmaPeriod, signalEmaPeriod)
        self.__bb = bollinger.BollingerBands(feed[instrument].getPriceDataSeries(), bBandsPeriod, 2)

        self.__verbose = verbose

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

    def get_start_price(self):
        return self.__startPrice

    def set_start_price(self, start_price):
        self.__startPrice = start_price

    def get_start_time(self):
        return self.__startTime

    def set_start_time(self, start_time):
        self.__startTime = start_time

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

    def get_share_values(self):
        return self.__shareValues

    def get_cprices(self):
        return self.__cprices

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
    def onBars(self, bars):
        n_shares = self.getBroker().getShares(self.get_instrument())
        bar = bars[self.get_instrument()]
        c_price = bar.getPrice()
        strat_cash = self.getBroker().getCash(False)

        delta_shares = int(total_cash*max_spend*sum([wt*st_onBars(self, bars) for wt, st_onBars in zip(self.__weights, self.__onBars)])/c_price)

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
            #print(history_onBars(self, bars))

        #print([st_onBars(self, bars) for st_onBars in self.__onBars])

        if not delta_shares == 0:
            self.marketOrder(self.__instrument, delta_shares)
            self.__cbasis = (n_shares*self.__cbasis + delta_shares*c_price) / (n_shares + delta_shares) if not n_shares + delta_shares == 0 else 0
        self.__portValues.append(self.getBroker().getEquity())
        self.__cprices.append(c_price)
        self.__shareValues.append(n_shares*c_price)

#best_model = [0.5240942184198141, 0.011991984061265401, 0.03332160740001963, 0.11681458821783589, 0.11902285924129916, 0.19005406060851363, 0.004700682051251965]
best_model = [0,0,0,0,0,0,0,0,1]
stock = sys.argv[1].lower()
period = sys.argv[2].lower() if len(sys.argv) >= 3 else "1y"
interval = sys.argv[3].lower() if len(sys.argv) >= 4 else ""

feed = quandlfeed.Feed()
if len(interval) == 0:
    feed.addBarsFromCSV(stock, "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower()))
else:
    feed.addBarsFromCSV(stock, "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), period.lower(), interval.lower()))

print("Winning model:",best_model)

myStrategy = MyStrategy(feed, stock, smaPeriod, best_model, verbose=True)
myStrategy.run()
print("Final portfolio value: $%.2f" % myStrategy.getBroker().getEquity())

plt.title("%s msp=%4.2f %s" % (stock.upper(), max_spend, str([str(x)[:4] for x in best_model])))
plt.plot([i for i in range(len(myStrategy.get_port_values()))], myStrategy.get_port_values(), label="Port Value")
#plt.show()
plt.plot([i for i in range(len(myStrategy.get_port_values()))], [p*int(total_cash/myStrategy.get_cprices()[0]) for p in myStrategy.get_cprices()], label="Adj Share Price")
plt.plot([i for i in range(len(myStrategy.get_port_values()))], myStrategy.get_share_values(), label="Port Value in Shares")
plt.legend()
plt.show()


