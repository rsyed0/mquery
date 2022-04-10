
import sys
from pyalgotrade.barfeed import quandlfeed

from single_strat import MyStrategy

def main():
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

    feed = quandlfeed.Feed()
    for stock, period in symbols:
        feed.addBarsFromCSV(stock, "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower()))

if __name__ == "__main__":
    main()