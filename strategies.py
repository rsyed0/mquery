# TODO: modify onBars methods to return ideal delta on standard scale (-1 to +1)
# where <0 corresponds to selling and >0 corresponds to buying
from pyalgotrade.technical import ma, rsi, macd, bollinger

from math import sqrt, sin, exp

total_cash = 10000

# strategy 1: SMA 
def sma_onBars(m_strategy, bars):
    # Wait for enough bars to be available to calculate a SMA.
    if m_strategy.get_sma()[-1] is None:
        return 0

    bar = bars[m_strategy.get_instrument()]
    n_shares = m_strategy.getBroker().getShares(m_strategy.get_instrument())
    c_price = bar.getPrice()
    sma = m_strategy.get_sma()[-1]

    p_diff = (c_price-sma) / c_price
    p_diff = min(1,max(-1,p_diff))
    return -sqrt(abs(p_diff)) if p_diff >= 0 else sqrt(abs(p_diff))

    """delta_shares = [10, -10]

    if sma < c_price:
    	return delta_shares[0]
    else:
    	return delta_shares[1]"""

# strategy 2: RSI
def rsi_onBars(m_strategy, bars):
    # Wait for enough bars to be available to calculate a SMA.
    if m_strategy.get_rsi()[-1] is None:
        return 0

    bar = bars[m_strategy.get_instrument()]
    n_shares = m_strategy.getBroker().getShares(m_strategy.get_instrument())
    c_price = bar.getPrice()
    rsi = m_strategy.get_rsi()[-1]

    p_diff = (50-rsi)/50
    p_diff = min(1,max(-1,p_diff))
    return -sqrt(abs(p_diff)) if p_diff >= 0 else sqrt(abs(p_diff))

    """cutoffs = [30, 50, 70]
    delta_shares = [10, 5, -5, -10]

    for i in range(len(cutoffs)):
    	if rsi < cutoffs[i]:
    		return delta_shares[i]

    return delta_shares[-1]"""

# strategy 1+2: SMA+RSI
def smarsi_onBars(m_strategy, bars):
    if m_strategy.get_sma()[-1] is None or m_strategy.get_rsi()[-1] is None:
        return 0

    # use SMA to inform RSI cutoffs (30/70 by default)
    bar = bars[m_strategy.get_instrument()]
    sma_diff = (bar.getPrice()-m_strategy.get_sma()[-1]) / bar.getPrice()
    dev_rsi = 200

    curr_rsi = m_strategy.get_rsi()[-1]

    reliability = 1-2.718**(-len(m_strategy.get_sma())/15.0)
    l_rsi, u_rsi = reliability*(30-sma_diff*dev_rsi), 100-(reliability*(100-(70-sma_diff*dev_rsi)))

    # modify buying amount from 10 shares based on strength of buying
    n_shares = m_strategy.getBroker().getShares(m_strategy.get_instrument())

    #d_rsi = l_rsi - curr_rsi if l_rsi > curr_rsi else (u_rsi - curr_rsi if u_rsi < curr_rsi else 0)
    p_diff = 0.5*(curr_rsi - l_rsi)/(u_rsi - l_rsi)
    
    p_diff = min(1,max(-1,p_diff))
    return -sqrt(abs(p_diff)) if p_diff >= 0 else sqrt(abs(p_diff))

# strategy 3: MACD
def macd_onBars(m_strategy, bars):
    if m_strategy.get_macd().getHistogram()[-1] is None:
        return 0

    # increase number of bought/sold shares with each day above/below 0
    bar = bars[m_strategy.get_instrument()]
    n_shares = m_strategy.getBroker().getShares(m_strategy.get_instrument())
    c_price = bar.getPrice()
    macd = m_strategy.get_macd().getHistogram()[-1]

    p_diff = macd / c_price
    p_diff = min(1,max(-1,p_diff))
    return abs(p_diff)**0.2 if p_diff >= 0 else -(abs(p_diff)**0.2)

    """delta_shares = [10, -10]

    if macd > 0:
    	return delta_shares[0]
    elif macd < 0:
    	return delta_shares[1]
    else:
    	return 0"""

# strategy 4: Bollinger bands
def bb_onBars(m_strategy, bars):
    if m_strategy.get_bb().getLowerBand()[-1] is None:
    	return 0

    bar = bars[m_strategy.get_instrument()]
    n_shares = m_strategy.getBroker().getShares(m_strategy.get_instrument())
    c_price = bar.getPrice()
    lower = m_strategy.get_bb().getLowerBand()[-1]
    upper = m_strategy.get_bb().getUpperBand()[-1]

    p_diff = 0.25*(c_price-lower)/(upper-lower)
    p_diff = min(1,max(-1,p_diff))
    #return -sqrt(abs(p_diff)) if p_diff >= 0 else sqrt(abs(p_diff))
    return -p_diff

    """delta_shares = [10, -10]

    if c_price < lower:
    	return delta_shares[0]
    elif c_price > upper:
    	return delta_shares[0]
    else:
    	return 0"""

# strategy 5: trend strategy (gap fill?)
# TODO consider tying output to trend duration as well
# TODO scrap this dogshit
def trend_onBars(m_strategy, bars):
    past_prices = m_strategy.get_cprices()
    bar = bars[m_strategy.get_instrument()]
    n_shares = m_strategy.getBroker().getShares(m_strategy.get_instrument())
    c_price = bar.getPrice()

    # TODO make these evolvable parameters
    retr_threshold, pct_threshold = 0.5, 0.15

    if len(past_prices) < 2:
        return 0
    elif len(past_prices) == 2:
        if past_prices[0] < past_prices[1]:
            m_strategy.set_last_bottom(past_prices[0])
            m_strategy.set_trend(1)
        else:
            m_strategy.set_last_top(past_prices[0])
            m_strategy.set_trend(-1)

    last_price = past_prices[-1]

    # when retracement hits threshold switch from uptrend to downtrend
    # keep track of last top and last bottom
    # avg change on up/downtrend <=> output
    last_top, last_bottom = m_strategy.get_last_top(), m_strategy.get_last_bottom()
    if m_strategy.get_trend() == -1:
        # downtrend, will have last top, may have last bottom
        if c_price > last_top:
            m_strategy.set_trend_length(0)
            m_strategy.set_last_bottom(None)
            m_strategy.set_last_top(c_price)
            return 0

        if last_bottom is not None:
            if c_price < last_bottom:
                m_strategy.set_last_bottom(None)
            elif (c_price - last_bottom) > retr_threshold*(last_top - last_bottom):
                # becomes an uptrend
                m_strategy.set_trend(1)
                m_strategy.set_trend_length(0)
                m_strategy.set_last_top(None)

                return 0
        elif c_price > last_price:
            m_strategy.set_last_bottom(last_price)

        trend_length = m_strategy.get_trend_length()+1
        m_strategy.set_trend_length(trend_length)

        # TODO compare (c_price - last_price) to avg movement
        avg_movement = (last_top - c_price) / m_strategy.get_trend_length()
        avg_pct_movement = avg_movement / c_price

        #print(last_top, c_price, (last_top - c_price) / c_price)

        # TODO implement capitulation (i.e., stop loss), perhaps thru sinusoidal willingness
        #willingness = 2.718**(m_strategy.get_trend_length() / 10)
        willingness = 3*exp(-trend_length/5)*sin(trend_length/2) #(25-((trend_length - 5)**2)) / 12.5
        return max(-1, min(1, avg_pct_movement * willingness)) #avg_pct_movement/pct_threshold)
    elif m_strategy.get_trend() == 1:
        # uptrend, will have last bottom, may have last top
        if c_price < last_bottom:
            m_strategy.set_trend_length(0)
            m_strategy.set_last_bottom(c_price)
            m_strategy.set_last_top(None)
            return 0

        if last_top is not None:
            if c_price > last_top:
                m_strategy.set_last_top(None)
            elif (last_top - c_price) > retr_threshold*(last_top - last_bottom):
                # becomes a downtrend
                m_strategy.set_trend(-1)
                m_strategy.set_trend_length(0)
                m_strategy.set_last_bottom(None)
                return 0
        elif c_price < last_price:
            m_strategy.set_last_top(last_price)

        trend_length = m_strategy.get_trend_length()+1
        m_strategy.set_trend_length(trend_length)

        avg_movement = (c_price - last_bottom) / m_strategy.get_trend_length()
        avg_pct_movement = avg_movement / c_price

        #print(last_bottom, c_price, (last_bottom - c_price) / c_price)

        willingness = 3*exp(-trend_length/5)*sin(trend_length/2)
        return max(-1, min(1, avg_pct_movement * willingness)) #-avg_pct_movement/pct_threshold)

    return 0

def normalize(model):
    # make so that all vals sum to 1
    s = sum(model)
    if s == 0:
        return [0 for x in model]
    return [x/s for x in model]

def history_onBars(m_strategy, bars):
    # TODO
    # calculate response of price to similar movements in the past
    # modulate/weight various time periods

    len_history = 15
    history = []

    bar = bars[m_strategy.get_instrument()]
    n_shares = m_strategy.getBroker().getShares(m_strategy.get_instrument())
    c_price = bar.getPrice()

    # check last few prices
    if len(m_strategy.get_cprices()) >= len_history:
        history = m_strategy.get_cprices()[-len_history:]
    else:
        return 0

    responses, weights = [], []
    for diff in range(1,len_history//2):
        pct_chg_i = (c_price - history[-diff]) / c_price

        for i in range(diff, len_history-diff-1):
            pct_chg_j = (history[i+diff] - history[i]) / history[i+diff]
            pct_diff = abs(pct_chg_i - pct_chg_j)

            response = (c_price - history[i+diff]) / c_price
            wt = max(0,1-2*sqrt(pct_diff))

            responses.append(response)
            weights.append(wt)

    weights = normalize(weights)
    wt_response = sum([wt*res for wt,res in zip(weights,responses)])

    #if len(m_strategy.get_cprices()) == 250:
    #    print(history, responses, weights, wt_response, sep="\n")

    out = 2*sqrt(wt_response) if wt_response >= 0 else -2*sqrt(-wt_response)
    return max(-1, min(1, out))

# strategy ?: support/resistance
def supres_onBars(m_strategy, bars):
	pass

# strategy 6: average down/cost-basis decision making
# TODO implement hesitation when not much cash left (can get trapped)
def cbasis_onBars(m_strategy, bars):
    bar = bars[m_strategy.get_instrument()]
    n_shares = m_strategy.getBroker().getShares(m_strategy.get_instrument())
    c_price = bar.getPrice()
    c_basis = m_strategy.get_cbasis()

    threshold = 0.025*c_basis

    strat_cash = m_strategy.getBroker().getCash(False)
    willingness = 2.718**((strat_cash-total_cash)/total_cash)

    p_diff = (c_price-c_basis) / c_price
    p_diff = min(1,max(-1,willingness*p_diff))
    return -sqrt(abs(p_diff)) if p_diff >= 0 else sqrt(abs(p_diff))

    """delta_shares = [5, -5]
    if n_shares < 0:
        if c_price > c_basis + threshold:
        	return delta_shares[0]#*willingness
        elif c_price < c_basis - threshold:
        	return delta_shares[1]#*willingness
    elif n_shares > 0:
        if c_price < c_basis - threshold:
            return delta_shares[0]#*willingness
        elif c_price > c_basis + threshold:
            return delta_shares[1]#*willingness
    else:
        return delta_shares[0]#*willingness

    return 0"""

# TODO optimize with heap for large len_history
def gfill_onBars(m_strategy, bars):
    len_history = 10
    history = []

    bar = bars[m_strategy.get_instrument()]
    n_shares = m_strategy.getBroker().getShares(m_strategy.get_instrument())
    c_price = bar.getPrice()

    # check last few prices
    if len(m_strategy.get_cprices()) >= len_history:
        history = m_strategy.get_cprices()[-len_history:]
    elif len(m_strategy.get_cprices()) > 0:
        history = m_strategy.get_cprices()
    else:
        return 0

    avg_price = sum(history)/len(history)
    min_price, max_price = min(history), max(history)
    
    if c_price > max_price:
        return max(-1, -0.5-2*sqrt((c_price-max_price)/max_price))
    elif c_price < min_price:
        return min(1, 0.5+2*sqrt((min_price-c_price)/min_price))

    if c_price > avg_price:
        return -0.5*(c_price-avg_price)/(max_price-avg_price)
    elif c_price < avg_price:
        return 0.5*(c_price-min_price)/(avg_price-min_price)

    return 0

# TODO output -1 to 1 based on price n_ints later
def nInts_idVal(self, prices, window_size=10, n_ints=25):
    pass

# strategy *: composite of working indicators (feedback loop)
# TODO figure out how to use this with theta strategies