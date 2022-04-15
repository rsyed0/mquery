import sys
from pyalgotrade.barfeed import quandlfeed

from pat_papertrade import WeightedIndicatorStrategy

pop_size = 30
n_generations = 10
n_survivors = 10
p_mutation = 0.5
msp_dominance = 0.5  # 0.5 to 1

allow_negation = False
n_models = 7*(2 if allow_negation else 1)

keep_best_model = False

class MultipleWeightedIndicatorStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instruments, weights):
        super(WeightedIndicatorStrategy, self).__init__(feed, 10000)

def normalize(model):
    # make so that all vals sum to 1
    s = sum(model)
    return [x/s for x in model]

def main():
    if len(sys.argv) < 2:
        sys.exit()

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

    if len(this_symbol) == 1:
        symbols.append((this_symbol[0], '1y'))
    elif len(this_symbol) == 2:
        symbols.append(this_symbol)

    feed = quandlfeed.Feed()
    for stock, period in symbols:
        feed.addBarsFromCSV(stock, "WIKI-%s-%s-yfinance.csv" % (stock.upper(), period.lower()))

    instruments = [s.lower() for s,_ in symbols]
    n_symbols = len(symbols)

    population = []
    for i in range(pop_size):
        pop_member = []
        for j in range(n_symbols):
            pop_member.append(normalize([random() for i in range(n_models)]))
        population.append(pop_member)

    this_strategy = MultipleWeightedIndicatorStrategy(feed, instruments, )

    print(symbols)

if __name__ == "__main__":
    main()