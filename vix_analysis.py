import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("VIX_History.csv")
dates = df["DATE"]
close_prices = df["CLOSE"].to_numpy()
print(close_prices)

# checking entries and exits of given range
#lower,upper = 20,35

for lower in range(15,21):
	for upper in range(lower+15,lower+21):

		mode = 1 if close_prices[0] < lower else (2 if close_prices[0] < upper else 3)
		entries,exits = [],[]

		for i in range(1,len(close_prices)):
			price = close_prices[i]
			this_mode = 1 if close_prices[i] < lower else (2 if close_prices[i] < upper else 3)

			if not mode == this_mode:
				# if skipped 2 entirely
				if mode+this_mode == 4:
					continue
				else:
					if mode == 2:
						exits.append((dates[i],this_mode))
					else:
						entries.append((dates[i],mode))
			mode = this_mode

		res_lower = 0
		peaks_between_troughs = []
		pbt = 0

		for date,mode in exits:
			if mode == 1:
				res_lower += 1
				peaks_between_troughs.append(pbt)
				pbt = 0
			else:
				pbt += 1

		print("resolving lower than "+str(lower)+": "+str(res_lower)+"/"+str(len(exits)))
		print("resolving higher than "+str(upper)+": "+str(len(exits)-res_lower)+"/"+str(len(exits)))
		print(peaks_between_troughs)

		plt.hist(peaks_between_troughs)
		plt.title("PBT with lower="+str(lower)+" and upper="+str(upper))
		plt.show()