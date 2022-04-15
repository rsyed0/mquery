# TODO cache values of all indicators from one run of strategy
class CachedWeightedIndicatorStrategy(strategy.BacktestingStrategy):
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

        self.__indicatorValues = None

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