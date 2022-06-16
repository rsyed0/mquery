from pyalgotrade import strategy, plotter
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.technical import ma, rsi, macd, bollinger

from pyalgotrade.bar import Frequency
from pyalgotrade.barfeed.csvfeed import GenericBarFeed

def get_stock_path(argv):
    stock = argv[1].lower()
    period = argv[2].lower() if len(argv) >= 3 else "1y"
    interval = argv[3].lower() if len(argv) >= 4 else ""

    stock_path = None
    if len(interval) == 0:
        stock_path = "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower())
    else:
        stock_path = "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), period.lower(), interval.lower())

    return stock_path, stock, period, interval

def get_stock_feed(stock, stock_path, interval=""):
    stock_feed = quandlfeed.Feed() if interval.lower() == "" else GenericBarFeed(Frequency.MINUTE)
    stock_feed.addBarsFromCSV(stock, stock_path)

    return stock_feed