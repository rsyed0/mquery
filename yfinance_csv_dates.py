import yfinance as yf

import sys

def fetch_price_csv(ticker_name, start_date, end_date, verbose=False):
    msft = yf.Ticker(ticker_name)

    # get stock info
    #print(msft.info)

    # get historical market data
    hist = msft.history(start=start_date, end=end_date)
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

    ymd_dates = [str(x)[:10] for x in list(hist_imp.index)] #if ticker_interval == "1d" else str(x)[:19]
    """ymd_times = ["" if ticker_interval == "1d" else str(x)[10:] for x in list(hist_imp.index)]

    nyd = []
    for i in range(len(ymd_dates)):
        print(ymd_dates[i])
        m,d,y = ymd_dates[i].split('/')
        nyd.append("%4s-%2s-%2s" % (y,m,d))

    ymd_dates = [d+t for d,t in zip(nyd, ymd_times)]"""

    hist_imp["Date"] = ymd_dates # if ticker_interval == "1d" else "Date Time"
    hist_imp.index = ymd_dates

    if verbose:
        print(hist_imp)

    hist_imp.to_csv("WIKI-%s-%s-%s-yfinance.csv" % (ticker_name, start_date, end_date))

if __name__ == "__main__":
    if not len(sys.argv) == 4:
        print("need more args")
        sys.exit(1)

    fetch_price_csv(sys.argv[1].upper(), sys.argv[2], sys.argv[3])