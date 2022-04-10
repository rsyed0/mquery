# strategy 1: SMA 
def sma_onBars(m_strategy, bars):
    # Wait for enough bars to be available to calculate a SMA.
    if m_strategy.get_sma()[-1] is None:
        return

    bar = bars[m_strategy.get_instrument()]
    # If a position was not opened, check if we should enter a long position.
    if m_strategy.get_position() is None:
        if bar.getPrice() > m_strategy.get_sma()[-1]:
            # Enter a buy market order for 10 shares. The order is good till canceled.
            m_strategy.set_position(m_strategy.enterLong(m_strategy.get_instrument(), 10, True))
    # Check if we have to exit the position.
    elif bar.getPrice() < m_strategy.get_sma()[-1] and not m_strategy.get_position().exitActive():
        m_strategy.get_position().exitMarket()

# strategy 2: RSI
def rsi_onBars(m_strategy, bars):
    # Wait for enough bars to be available to calculate a SMA.
    if m_strategy.get_rsi()[-1] is None:
        return

    bar = bars[m_strategy.get_instrument()]
    # If a position was not opened, check if we should enter a long position.
    if m_strategy.get_position() is None:
        if 30 > m_strategy.get_rsi()[-1]:
            # Enter a buy market order for 10 shares. The order is good till canceled.
            m_strategy.set_position(m_strategy.enterLong(m_strategy.get_instrument(), 10, True))
    # Check if we have to exit the position.
    elif 70 < m_strategy.get_rsi()[-1] and not m_strategy.get_position().exitActive():
        m_strategy.get_position().exitMarket()

# strategy 1+2: SMA+RSI
# TODO: fix issues with buying past cash allotment, RSI at start of window, no stop loss, etc
def smarsi_onBars(m_strategy, bars):
    if m_strategy.get_sma()[-1] is None or m_strategy.get_rsi()[-1] is None:
        return

    # use SMA to inform RSI cutoffs (30/70 by default)
    bar = bars[m_strategy.get_instrument()]
    sma_diff = (bar.getPrice()-m_strategy.get_sma()[-1]) / bar.getPrice()
    dev_rsi = 200

    curr_rsi = m_strategy.get_rsi()[-1]

    reliability = 1-2.718**(-len(m_strategy.get_sma())/15.0)
    l_rsi, u_rsi = reliability*(30-sma_diff*dev_rsi), 100-(reliability*(100-(70-sma_diff*dev_rsi)))

    # modify buying amount from 10 shares based on strength of buying
    n_shares = m_strategy.getBroker().getShares(m_strategy.get_instrument())
    delta_n_shares = l_rsi - curr_rsi if l_rsi > curr_rsi else (u_rsi - curr_rsi if u_rsi < curr_rsi else 0)
    
    #delta_n_shares *= reliability

    #if delta_n_shares > 0:
        #m_strategy.set_position(m_strategy.enterLong(m_strategy.get_instrument(), delta_n_shares, True))
    if not int(delta_n_shares) == 0:
        print("Buying" if delta_n_shares > 0 else "Selling",int(abs(delta_n_shares)),"shares at",bar.getPrice())
    m_strategy.marketOrder(m_strategy.get_instrument(), int(delta_n_shares))

    # If a position was not opened, check if we should enter a long position.
    """if abs(delta_n_shares) > 0 and abs(n_shares+int(delta_n_shares)) > 0:
        if delta_n_shares > 0:
            # Enter a buy market order for 10 shares. The order is good till canceled.
            #m_strategy.marketOrder(m_strategy.get_instrument(), 10)
            #m_strategy.get_position().exitMarket()

            m_strategy.set_position(m_strategy.enterLong(m_strategy.get_instrument(), n_shares+int(delta_n_shares), True))
        # Check if we have to exit the position.
        elif delta_n_shares < 0 and n_shares+int(delta_n_shares) > 0:
            if m_strategy.get_position() is not None:
                m_strategy.get_position().exitMarket()
            m_strategy.set_position(m_strategy.enterLong(m_strategy.get_instrument(), n_shares+int(delta_n_shares), True))"""

# strategy 3: MACD
def macd_onBars(m_strategy, bars):
    if m_strategy.get_macd().getHistogram()[-1] is None:
        return

    # TODO increase number of bought/sold shares with each day above/below 0

    if m_strategy.get_position() is None:
        if m_strategy.get_macd().getHistogram()[-1] > 0:
            m_strategy.set_position(m_strategy.enterLong(m_strategy.get_instrument(), 10, True))
    elif m_strategy.get_macd().getHistogram()[-1] < 0 and not m_strategy.get_position().exitActive():
        m_strategy.get_position().exitMarket()

# strategy 4: Bollinger bands
def bb_onBars(m_strategy, bars):
    lower = self.__bbands.getLowerBand()[-1]
    upper = self.__bbands.getUpperBand()[-1]

    shares = self.getBroker().getShares(self.__instrument)
    bar = bars[self.__instrument]
    if shares == 0 and bar.getClose() < lower:
        sharesToBuy = int(self.getBroker().getCash(False) / bar.getClose())
        self.info("Placing buy market order for %s shares" % sharesToBuy)
        self.marketOrder(self.__instrument, sharesToBuy)
    elif shares > 0 and bar.getClose() > upper:
        self.info("Placing sell market order for %s shares" % shares)
        self.marketOrder(self.__instrument, -1*shares)

# strategy 5: VIX
# strategy 6: support/resistance
# strategy 7: average down/cost-basis decision making
def cbasis_onBars(m_strategy, bars):
    pass

# strategy *: composite of working indicators (feedback loop)
# how to use this with theta strategies