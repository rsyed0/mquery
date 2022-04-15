from pat_papertrade import *
from strategies import *

import sys, os
import matplotlib.pyplot as plt

total_cash = 10000

def main():
    period = sys.argv[1]
    interval = sys.argv[2] if len(sys.argv) > 2 else "1d"

    filenames = os.listdir(os.getcwd())
    applicable = []
    for fn in filenames:
        if not fn[:4] == "WIKI":
            continue

        tokens = fn.split('-')
        if len(tokens) < 4:
            continue

        if tokens[2] == "USD":
            tokens[1] += "USD"
            tokens.pop(2)

        symbol, pd, invl = tokens[1:4]

        if invl == "yfinance.csv":
            invl = "1d"

        #print(symbol, pd, period, invl, interval)
        if pd == period and invl == interval:
            applicable.append(fn)

    print(applicable)
    input("Will run genetic algorithm for %s symbols. Press enter to continue." % (len(applicable)))

    scores = []
    for fn in applicable:
        # use genetic algorithm with filename
        # TODO switch to RNN-based algorithm if performance is better

        tokens = fn.split("-")

        if tokens[2] == "USD":
            tokens[1] += "USD"
            tokens.pop(2)

        symbol, pd, invl = tokens[1:4]
        if invl == "yfinance.csv":
            invl = ""

        print("running",symbol)
        symbol = symbol.lower()

        wi_strategy, max_spend = run_strategy(symbol, pd, invl, 15, 5, False)
        share_value = wi_strategy.getBroker().getShares(wi_strategy.get_instrument())*wi_strategy.get_cprices()[-1]
        total_value = wi_strategy.getBroker().getEquity()
        exposure = share_value / total_value
        rating = sum(wi_strategy.get_model_wt_values())

        ovr_performance = wi_strategy.get_cprices()[-1] * (total_cash / wi_strategy.get_cprices()[0])
        mkt_diff = (total_value - ovr_performance) / total_cash

        scores.append((symbol, exposure, rating, mkt_diff))

    scores.sort(key=lambda x:x[2], reverse=True)
    print(scores)

    labels = [x[0] for x in scores]
    exposures = [x[1] for x in scores]
    ratings = [x[2] for x in scores]

    #fig, ax = plt.subplots()
    plt.scatter(exposures, ratings)
    plt.xlabel("Exposure of Best Classifier (h)")
    plt.ylabel("Buy/Sell Rating of Best Classifier (dh/dt)")

    dot_diameter = 0.005
    for i,label in enumerate(labels):
        plt.annotate(label.upper(), (exposures[i]+dot_diameter, ratings[i]+dot_diameter))

    plt.show()
    plt.clf()

    performance = [x[3] for x in scores]
    bullishness = [e+s for e,s in zip(exposures, ratings)]
    plt.scatter(performance, bullishness)
    plt.xlabel("Performance of Classifier (pct over default)")
    plt.ylabel("Bullishness of Classifier (h + dh/dt)")

    dot_diameter = 0.005
    for i,label in enumerate(labels):
        plt.annotate(label.upper(), (performance[i]+dot_diameter, bullishness[i]+dot_diameter))

    plt.show()


if __name__ == "__main__":
    main()