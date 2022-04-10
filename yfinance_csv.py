import yfinance as yf

import sys

ticker_name = sys.argv[1].upper()
ticker_period = sys.argv[2].lower() if len(sys.argv) >= 3 else "1y"
ticker_interval = sys.argv[3].lower() if len(sys.argv) >= 4 else "1d"

msft = yf.Ticker(ticker_name)

# get stock info
#print(msft.info)

# get historical market data
hist = msft.history(period=ticker_period, interval=ticker_interval)
print(hist)
hist_imp = hist[["Open","High","Low","Close","Volume"]]

# append Adj Close (just keep same as close)
hist_imp["Adj. Close"] = hist_imp["Close"]

# change dates to y-m-d
# can also do via excel
ymd_dates = [str(x)[:10] for x in list(hist_imp.index)] if ticker_interval == "1d" else [str(x)[:19] for x in list(hist_imp.index)]
hist_imp["Date" if ticker_interval == "1d" else "Date Time"] = ymd_dates
hist_imp.index = ymd_dates

print(hist_imp)

if ticker_interval == "1d":
	hist_imp.to_csv("WIKI-%s-%s-yfinance.csv" % (ticker_name, ticker_period))
else:
	hist_imp.to_csv("WIKI-%s-%s-%s-yfinance.csv" % (ticker_name, ticker_period, ticker_interval))