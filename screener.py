from pat_papertrade import *
from strategies import *

import sys, os
import matplotlib.pyplot as plt

from datetime import date

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
    for i in range(len(applicable)):
        # use genetic algorithm with filename
        # TODO switch to RNN-based algorithm if performance is better
        fn = applicable[i]
        tokens = fn.split("-")

        if tokens[2] == "USD":
            tokens[1] += "USD"
            tokens.pop(2)

        symbol, pd, invl = tokens[1:4]
        if invl == "yfinance.csv":
            invl = ""

        print("Running %s (%d/%d)" % (symbol, i+1, len(applicable)))
        symbol = symbol.lower()

        wi_strategy, max_spend = run_strategy(symbol, pd, invl, 15, 5, False)
        share_value = wi_strategy.getBroker().getShares(wi_strategy.get_instrument())*wi_strategy.get_cprices()[-1]
        total_value = wi_strategy.getBroker().getEquity()
        exposure = share_value / total_value
        rating = sum(wi_strategy.get_model_wt_values())

        ovr_performance = wi_strategy.get_cprices()[-1] * (total_cash / wi_strategy.get_cprices()[0])
        mkt_diff = (total_value - ovr_performance) / total_cash

        scores.append((symbol, exposure, rating, mkt_diff, max_spend, wi_strategy))

    scores.sort(key=lambda x:x[2], reverse=True)
    #print(scores)

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
    bullishness = [e+x[4]*s for e,s,x in zip(exposures, ratings, scores)]
    plt.scatter(performance, bullishness)
    plt.xlabel("Performance of Classifier (pct over default)")
    plt.ylabel("Bullishness of Classifier (h + spend*dh/dt)")

    for i,label in enumerate(labels):
        plt.annotate(label.upper(), (performance[i]+dot_diameter, bullishness[i]+dot_diameter))

    plt.show()

    # save all models to screener_models_050222.txt
    today = date.today()
    with open('screener_models_%s.txt' % (today.strftime('%m%d%y')), 'w') as f:
        for sym,_,_,_,msp,model in scores:
            f.write("%s:%s, %4.2f" % (sym,str(model.get_weights()),msp))
        f.close()

    # allow user to examine models for certain stocks
    u_input = ""
    sym_to_model = {sym:model for sym,_,_,_,_,model in scores}

    while not u_input == "!q":
        u_input = input("Enter symbol(s) to examine: ")
        tokens = u_input.split(" ")

        if len(tokens) == 1:
            stock = tokens[0].lower()
            if stock in sym_to_model:
                model = sym_to_model[stock]
                model_weights = model.get_weights()
                model_max_spend = model.get_max_spend()

                feed = quandlfeed.Feed() if interval.lower() == "1d" else GenericBarFeed(Frequency.MINUTE)
                if interval == "1d":
                    feed.addBarsFromCSV(stock, "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower()))
                else:
                    feed.addBarsFromCSV(stock, "WIKI-%s-%s-%s-yfinance.csv" % (stock.upper(), period.lower(), interval.lower()))

                print(model_weights, model_max_spend)

                myStrategy = WeightedIndicatorStrategy(feed, stock, model_weights, max_spend=model_max_spend)
                myStrategy.run()

                print("Ending score: $%8.2f" % (myStrategy.get_port_values()[-1]))

                plt.title(stock.upper()) #str([str(x)[:4] for x in best_model]))
                plt.plot([i for i in range(len(myStrategy.get_port_values()))], myStrategy.get_port_values(), label="Port Value")
                plt.plot([i for i in range(len(myStrategy.get_port_values()))], [p*int(total_cash/myStrategy.get_cprices()[0]) for p in myStrategy.get_cprices()], label="Adj Share Price")
                plt.plot([i for i in range(len(myStrategy.get_port_values()))], myStrategy.get_share_values(), label="Port Value in Shares")
                plt.legend()
                plt.show()
            elif stock == "-ls":
                print([x[0] for x in scores])
            elif not stock == "!q":
                print("Invalid symbol")
        elif len(tokens) == 2:
            # cross-apply tokens[0] classifier to tokens[1]
            # TODO reduce repetition, combine this w/ prev clause
            s1, s2 = [x.lower() for x in tokens]
            if s1 in sym_to_model and s2 in sym_to_model:
                model = sym_to_model[s1]
                model_weights = model.get_weights()
                model_max_spend = model.get_max_spend()

                feed = quandlfeed.Feed() if interval.lower() == "1d" else GenericBarFeed(Frequency.MINUTE)
                if interval == "1d":
                    feed.addBarsFromCSV(s2, "WIKI-%s-%s-yfinance.csv" % (s2.upper(), period.lower()))
                else:
                    feed.addBarsFromCSV(s2, "WIKI-%s-%s-%s-yfinance.csv" % (s2.upper(), period.lower(), interval.lower()))

                print(model_weights, model_max_spend)

                myStrategy = WeightedIndicatorStrategy(feed, s2, model_weights, max_spend=model_max_spend)
                myStrategy.run()

                print("Ending score: $%8.2f" % (myStrategy.get_port_values()[-1]))

                plt.title("%s using %s classifier" % (s2.upper(), s1.upper())) #str([str(x)[:4] for x in best_model]))
                plt.plot([i for i in range(len(myStrategy.get_port_values()))], myStrategy.get_port_values(), label="Port Value")
                plt.plot([i for i in range(len(myStrategy.get_port_values()))], [p*int(total_cash/myStrategy.get_cprices()[0]) for p in myStrategy.get_cprices()], label="Adj Share Price")
                plt.plot([i for i in range(len(myStrategy.get_port_values()))], myStrategy.get_share_values(), label="Port Value in Shares")
                plt.legend()
                plt.show()

            else:
                print("Invalid symbol")
        else:
            print("Invalid number of tokens")

if __name__ == "__main__":
    main()