import yfinance as yf

import sys

def fetch_price_csv(ticker_name=None, ticker_period=None, ticker_interval=None, verbose=False):
    if ticker_name is None or ticker_period is None or ticker_interval is None:
        ticker_name = sys.argv[1].upper()
        ticker_period = sys.argv[2].lower() if len(sys.argv) >= 3 else "1y"
        ticker_interval = sys.argv[3].lower() if len(sys.argv) >= 4 else "1d"

    msft = yf.Ticker(ticker_name)

    # get stock info
    #print(msft.info)

    # get historical market data
    hist = msft.history(period=ticker_period, interval=ticker_interval)
    hist_imp = hist[["Open","High","Low","Close","Volume"]]

    # append Adj Close (just keep same as close)
    hist_imp["Adj. Close"] = hist_imp["Close"]

    # ensure no low > close, high < close, etc
    for index, row in hist_imp.iterrows():
        for cr in ("Open", "Close"):
            if row[cr] >= row["High"]:
                row["High"] = row[cr] + 0.0001
            elif row[cr] <= row["Low"]:
                row["Low"] = row[cr] - 0.0001
        hist_imp.loc[index] = row

    # opening in excel auto changes to wrong date format...
    # just don't open in excel. excel is evil

    ymd_dates = [str(x)[:10] if ticker_interval == "1d" else str(x)[:19] for x in list(hist_imp.index)]
    """ymd_times = ["" if ticker_interval == "1d" else str(x)[10:] for x in list(hist_imp.index)]

    nyd = []
    for i in range(len(ymd_dates)):
        print(ymd_dates[i])
        m,d,y = ymd_dates[i].split('/')
        nyd.append("%4s-%2s-%2s" % (y,m,d))

    ymd_dates = [d+t for d,t in zip(nyd, ymd_times)]"""

    hist_imp["Date" if ticker_interval == "1d" else "Date Time"] = ymd_dates
    hist_imp.index = ymd_dates

    if verbose:
        print(hist_imp)

    if ticker_interval == "1d":
        hist_imp.to_csv("WIKI-%s-%s-yfinance.csv" % (ticker_name, ticker_period))
    else:
        hist_imp.to_csv("WIKI-%s-%s-%s-yfinance.csv" % (ticker_name, ticker_period, ticker_interval))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("need more args")
        sys.exit(1)

    if sys.argv[1] == '*':
        stocks = [s.upper() for s in sys.argv[2:-2]]
        period = sys.argv[-2].lower()
        interval = sys.argv[-1].lower()

        for stock in stocks:
            print("fetching", stock, "prices")
            fetch_price_csv(stock, period, interval)
    else:
        fetch_price_csv()