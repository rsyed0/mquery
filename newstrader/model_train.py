import sys, os
import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_collection import *
import model_train_utils as mtu

sys.path.append("../")
from yfinance_csv import fetch_price_csv

default_time_period = "1mo"
default_time_interval = "15m"
default_time_interval_mins = 15
default_time_interval_value = default_time_interval_mins*60
default_time_interval_to_day_value = int(6.5*60/default_time_interval_mins)
est_time_adj = 4*60*60

pct_threshold = 0.05
batch_size = 64
fwd_window_size = 5

from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# TODO train transformer encoders for sentiment analysis
# TODO use pre-scraped data in CSV if specified in args

# Assumed tweet dataframe schema:
# 	user: str - handle of user posting the tweet
#	time: long - Unix timestamp of tweet posting
# 	text: str - full text of tweet

def extract_stock_mentions(tweet_text):
	return [token[1:] for token in tweet_text.split(" ") if len(token) >= 2 and len(token) <= 6 and token[0] == "$" and token[1:].isupper() and token[1:].isalpha()]

def date_time_to_unix(date_time):
	date_format = datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
	return datetime.datetime.timestamp(date_format)

def main():

	tweets_df = None
	if len(sys.argv) > 1:
		fname = sys.argv[1]
		assert fname[-4:] == ".csv"
		tweets_df = pd.read_csv(fname)
	else:
		# TODO use tweepy api thru data_collection module
		pass

	user_to_tweets, stock_to_tweets = {}, {}
	for i, tweet in tweets_df.iterrows():
		if tweet["user"] in user_to_tweets:
			user_to_tweets[tweet["user"]].append(tweet)
		else:
			user_to_tweets[tweet["user"]] = [tweet]

		mentions = extract_stock_mentions(tweet["text"])
		for stock in mentions:
			if stock in stock_to_tweets:
				stock_to_tweets[stock].append(tweet)
			else:
				stock_to_tweets[stock] = [tweet]

	# can see all stocks mentioned with stock_to_tweets.keys()
	stock_to_prices = {}
	stock_to_train_data, stock_to_valid_data = {}, {}

	for stock in stock_to_tweets.keys():
		price_df = fetch_price_csv(stock.upper(), default_time_period, default_time_interval, save=False)

		# TODO convert ["Date Time"] into Unix timestamp
		price_df["Time"] = price_df["Date Time"].apply(date_time_to_unix)
		stock_to_prices[stock] = price_df
		print(price_df)

		start_time = price_df["Time"].values[0] - est_time_adj
		end_time = price_df["Time"].values[-1] - est_time_adj

		X_train, y_train = [], []
		for tweet in stock_to_tweets[stock]:
			# TODO generate y_train based on stock response to each tweet
			tweet_time = tweet["time"]

			if tweet_time < start_time or tweet_time > end_time:
				continue

			# TODO figure out a better way to do this
			print(tweet_time, start_time)
			price_df_index = None
			for i, time in enumerate(price_df["Time"].values):
				if time > tweet_time:
					price_df_index = max(0, i-1)
					break

			if not price_df_index:
				continue

			#price_df_index = int(((tweet_time - start_time) // default_time_interval_value) + 1)
			#print(price_df_index)
			start_price = price_df["Close"].values[price_df_index]
			end_price = price_df["Close"].values[min(len(price_df) - 1, price_df_index + fwd_window_size)]

			X_train.append(tweet["text"])
			y_train.append((end_price - start_price) / (start_price * pct_threshold))

		X_train, y_train, X_valid, y_valid = train_test_split(X_train, y_train)

		X_train_inputs, X_train_masks = mtu.preprocessing_for_bert(X_train)
		X_valid_inputs, X_valid_masks = mtu.preprocessing_for_bert(X_valid)

		# TODO determine if y-labels are of compatible format
		X_train_data = TensorDataset(X_train_inputs, X_train_masks, y_train)
		X_train_sampler = RandomSampler(X_train_data)

		X_valid_data = TensorDataset(X_valid_inputs, X_valid_masks, y_valid)
		X_valid_sampler = RandomSampler(X_valid_data)

		stock_to_train_data[stock] = DataLoader(X_train_data, sampler=X_train_sampler, batch_size=batch_size) 
		stock_to_valid_data[stock] = DataLoader(X_valid_data, sampler=X_valid_sampler, batch_size=batch_size)

	# TODO fetch market data using fetch_price_csv()
	# TODO examine impact on market in given time period
	# TODO fine tune BertModel from transformers
	stock_to_model = {}
	for stock, tweets in stock_to_tweets.items():
		"""X_train, y_train, X_valid, y_valid = train_test_split(stock_to_X_train[stock], stock_to_y_train[stock])

		# TODO initialize train/val_dataloader
		training_data = torch.Tensor([[xt, yt] for xt, yt in zip(X_train, y_train)])
		valid_data = torch.Tensor([[xv, yv] for xv, yv in zip(X_valid, y_valid)])"""

		train_dataloader, val_dataloader = stock_to_train_data[stock], stock_to_valid_data[stock]

		bert_classifier, optimizer, scheduler = mtu.initialize_model(epochs=5)
		mtu.train(bert_classifier, train_dataloader, val_dataloader, epochs=5, evaluation=True)

		stock_to_model[stock] = bert_classifier

	# Assumed equal weighting for each source
	# TODO decide whether to weight based on source account
	# will have to separate data based on source instead of stock


if __name__ == "__main__":
	main()