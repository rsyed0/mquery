from pat_papertrade import MyStrategy

from pyalgotrade import strategy, plotter
from pyalgotrade.barfeed import quandlfeed
from pyalgotrade.bitstamp import barfeed, broker
#from pyalgotrade.technical import ma, rsi, macd, bollinger

import matplotlib.pyplot as plt

smaPeriod = 15

def main():
    stock = "BTC" #sys.argv[1].lower()
    period = 0 #sys.argv[2].lower()

    # get live data feed
    barFeed = barfeed.LiveTradeFeed()
    brk = broker.PaperTradingBroker(1000, barFeed)

    best_model = [0,0,0,0,1]

    myStrategy = MyStrategy(barFeed, stock, smaPeriod, best_model, verbose=True)
    myStrategy.run()
    print("Final portfolio value: $%.2f" % myStrategy.getBroker().getEquity())

    plt.title(stock.upper()+" "+str([str(x)[:4] for x in best_model]))
    plt.plot([i for i in range(len(myStrategy.get_port_values()))], myStrategy.get_port_values())
    #plt.show()
    plt.plot([i for i in range(len(myStrategy.get_port_values()))], [p*int(total_cash/myStrategy.get_cprices()[0]) for p in myStrategy.get_cprices()])
    plt.show()

if __name__ == "__main__":
    main()
