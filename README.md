# mquery

## Setup:

python3 -m pip install pyalgotrade

python3 -m pip install keras

python3 -m pip install yfinance

- Requires Python 3 installation, not tested with Python 2
- Alternatively can use pip3

## Existing features:

- Wrapper for fetching yfinance ticker history (yfinance_csv.py)
- Implementation of several trading indicators (strategies.py)
- Implementation of genetic algorithm to find optimal weighting of these indicators (pat_papertrade.py)
	- Snapshots of this algorithm's optimal model for certain tickers and timeframes (highlights/, recents/)
- Implementation of recurrent neural network (rnn_algotrade.py)
	- Works best with sectors that display repeated patterns (e.g. energy)

## Planned features:

- Usage of quanttrader API to use live data/perform live trading
- Implementation of higher leverage trading strategies like calls/puts
- Hierarchical forecasting for stocks grouped by sector
- Web/front-end interface for stock analysis using Flask or Django
- Segmented RNN approach (volatility and bullishness), different normalizations

## Usage:

python3 yfinance_csv.py [symbol] [period=1y] [interval=1d]

python3 yfinance_csv.py * [symbols...] [period] [interval]
- Fetches stock price of symbol on period at given interval using yfinance API
- Cleans data of possible issues (high < close/open, low > close/open)
- Can also fetch multiple symbols using * token

python3 single_strat.py [symbol] [period=1y] [interval=1d]
- Weighted-average of several indicators, based on weights/indicators set in code
- Generates line plot to show time-series exposure and performance of algorithm on period
- Useful for testing performance of single classifier, especially when creating custom

python3 pat_algotrade.py [symbol] [period=1y] [interval=1d]
- Weighted-average of several indicators, tuned using genetic algorithm
- Generates line plot to show time-series exposure and performance of algorithm on period
- Can be customized by adding strategies to strategies.py

python3 rnn_algotrade.py [symbol] [train_start] [train_end] [test_start] [test_end]
- LSTM RNN model trained to use indicator outputs as input
- Generates line plot to show time-series exposure and performance of algorithm on period
- Aiming to predict weighted future performance of stock on next 25 trading periods

python3 screener.py [period] [interval]
- Screen all stocks in local directory with given period and interval
- Uses genetic algorithm to determine bullishness at end of period
- Generates two scatter plots (exposure vs buy/sell, performance vs bullishness)
- Allows user to examine line plot and weights for each stock/classifier performance 
- Saves each classifier in a file called screener_[mmddyy].txt

python3 multiple_series.py [symbols...] [period=1y] [interval=1d]
- Weighted-average of several indicators over several symbols, tuned using genetic algorithm
- Will accumulate shares of multiple symbols within same portfolio to maximize returns
- Generates line plot to show time-series exposure and performance of algorithm on period
- Can be customized by adding strategies to strategies.py