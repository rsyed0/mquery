mquery

Setup:
- pip install pyalgotrade
- pip install keras

Existing features:
- Wrapper for fetching yfinance ticker history (yfinance_csv.py)
- Implementation of several trading indicators (strategies.py)
- Implementation of genetic algorithm to find optimal weighting of these indicators (pat_papertrade.py)
	- Snapshots of this algorithm's optimal model for certain tickers and timeframes (highlights, recents)

Planned features:
- Implementation of recurrent neural network (rnn_algotrade.py)
- Implementation of multiple-ticker algotrading strategy
- Usage of quanttrader API to use live data/perform live trading
