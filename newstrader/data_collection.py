import tweepy
import pandas as pd
import numpy as np
import datetime

# TODO first try training model with pre-scraped data

user_ids = ["Fxhedgers", "Deltaone", "unusual_whales", "zerohedge", "breakingmkts", "FirstSquawk"]

# TODO figure out if this has any usages
bearer_token = "AAAAAAAAAAAAAAAAAAAAAOj4eQEAAAAAWRenITqFCtY1mQAcl09Lfy%2Fii5U%3DnL3TIAi3giT29vpqspSojgyqpTObti01YBDj0oFtcCo0UEjXTW"

def create_auth(consumer_key, consumer_secret, access_token, access_token_secret):
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)

	return auth, api

def main():
	# TODO before pushing move these to a file ignored by Git
	api_key, api_key_secret, access_token, access_token_secret = None, None, None, None

	try:
		api_key, api_key_secret, access_token, access_token_secret = [str(x) for x in open("api_keys.txt", "r").readlines()]
	except FileNotFoundError:
		print("Need to set up API keys in ./api_keys.txt")
		sys.exit(1)
	except Exception:
		print("./api_keys.txt should have exactly 4 lines")
		sys.exit(1)

	# TODO get/authorize Twitter/Tweepy app credentials
	auth, api = create_auth(api_key, api_key_secret, access_token, access_token_secret)
	tweets_arr = []

	for user_id in user_ids:
		# collect tweets and timestamps using Twitter API
		tweets = api.user_timeline(screen_name=user_id, 
		                           # 200 is the maximum allowed count
		                           count=200,
		                           include_rts = False,
		                           # Necessary to keep full_text 
		                           # otherwise only the first 140 words are extracted
		                           tweet_mode = 'extended'
		                           )
		
		for tweet in tweets:
			timestamp = datetime.datetime.timestamp(datetime.datetime.strptime(str(tweet.created_at)[:-6], "%Y-%m-%d %H:%M:%S"))
			tweets_arr.append([user_id, timestamp, tweet.full_text])

		#print(tweets[0].full_text)

	# TODO save last 200 tweets to CSV
	tweets_df = pd.DataFrame(np.array(tweets_arr), columns=["user", "time", "text"])
	tweets_df.to_csv("tweets.csv")

if __name__ == "__main__":
	main()
