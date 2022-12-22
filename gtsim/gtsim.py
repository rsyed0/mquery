from bisect import bisect_left
from random import random
from math import exp
from gtsim_strategies import *
from pyalgotrade import strategy

import pandas as pd

# game theory-based market sim between x actors
# reinforcement learning based on P/L with real-time bid-ask spread
# decide on inner model (window-based, indicators, etc?)
# and output structure (buy/sell indicator?)
# also decide if actor can cancel orders, partial execution, etc

PRICE_DENOM = 1.0
WINDOW_SIZE = 25
STARTING_PRICE = 35.0
STARTING_CASH = 10000.0
MARGIN_INTEREST = 0.0
SHORT_FEE = 0.0
BUY_SELL_THRESHOLD = 0
SPENDING_MULTIPLIER = 1000.0

# sim vars
N_ACTORS = 25
SIM_LENGTH = 250

class Actor:
    def __init__(self, sim, actor_id, params=None, start_cash=None):
        self.sim = sim
        self.cash_on_hand = start_cash if start_cash else STARTING_CASH
        self.n_shares = 0
        self.orders = {}
        self.actor_id = actor_id

        # TODO implement cost basis

        # map "max_spend", "window_mult", "spread_mult"
        if not params:
            params = {}
            params["max_spend"] = random()*2-1
            params["window_mult"] = [random()*2-1 for i in range(WINDOW_SIZE-1)]
            params["spread_mult"] = random()*2-1
        self.params = params

        self.n_shares_hist, self.cash_hist = [], []

    def model(self, window, bid_quote, ask_quote):
        # TODO find way to integrate n_shares for bid/ask quotes
        # TODO apply ../strategies.py here
        if not bid_quote and not ask_quote:
            mid_quote = window[-1]
        elif not bid_quote:
            mid_quote = ask_quote[0]
        elif not ask_quote:
            mid_quote = bid_quote[0]
        else:
            mid_quote = (bid_quote[0]+ask_quote[0])/2

        window_pct_diffs = [(window[i+1]-window[i])/window[i] for i in range(len(window)-1)]
        spread_pct_diff = (mid_quote-window[-1])/window[-1]
        weighted_pct_diffs = sum([x*y for x,y in zip(window_pct_diffs, self.params["window_mult"])]) + spread_pct_diff*self.params["spread_mult"]

        # TODO implement weighting for each part of window/spread

        # return logistic to coerce to -1,1
        out = 2/(1+exp(-weighted_pct_diffs)) - 1
        print(out)
        return out

    def execute_step(self):
        window = self.sim.last_prices[-min([len(self.sim.last_prices), WINDOW_SIZE]):]
        bid_quote, ask_quote = self.sim.quote_bid(), self.sim.quote_ask()

        # run model, convert to order(s), submit/log order(s)
        buy_sell_ind = self.model(window, bid_quote, ask_quote)

        if abs(buy_sell_ind) > BUY_SELL_THRESHOLD:
            # buy_sell_ind b/w -1 and +1
            # -1 => sell at any price possible (at bid)
            # 1 => buy at any price possible (at ask)

            bid_price = bid_quote[0] if bid_quote else self.sim.last_prices[-1] - PRICE_DENOM
            ask_price = ask_quote[0] if ask_quote else self.sim.last_prices[-1] + PRICE_DENOM
            #assert bid_price < ask_price

            mid_price = (bid_price + ask_price) / 2
            order_price = ((mid_price + (ask_price - bid_price) * buy_sell_ind / 2) // PRICE_DENOM) * PRICE_DENOM

            if order_price <= 0:
                order_price = PRICE_DENOM

            money_vol = abs(buy_sell_ind) * SPENDING_MULTIPLIER
            order_n_shares = int(money_vol / order_price)

            if order_n_shares > 0:
                order_id = None
                if buy_sell_ind > 0:
                    order_id = self.sim.submit_buy(order_n_shares, order_price, self.actor_id)
                else:
                    order_id = self.sim.submit_sell(order_n_shares, order_price, self.actor_id)

                self.orders[order_id] = (buy_sell_ind > 0, order_n_shares, order_price)

        self.n_shares_hist.append(self.n_shares)
        self.cash_hist.append(self.cash_on_hand)

        # apply margin/shorting penalties
        if self.cash_on_hand < 0:
            self.cash_on_hand -= abs(self.cash_on_hand) * MARGIN_INTEREST
        if self.n_shares < 0:
            self.cash_on_hand -= abs(self.n_shares) * SHORT_FEE

    def calculate_pnl(self):
        pnl = self.n_shares * self.sim.last_prices[-1] + self.cash_on_hand
        return pnl

    def notify(self, order_id, exec_n_shares, exec_price=None):
        is_buy, order_n_shares, price = self.orders[order_id]

        if not exec_price:
            exec_price = price

        if is_buy:
            self.cash_on_hand -= exec_n_shares * exec_price
            self.n_shares += exec_n_shares
        else:
            self.cash_on_hand += exec_n_shares * exec_price
            self.n_shares -= exec_n_shares


class Stock:
    def __init__(self, actors=None, last_prices=None):
        self.bid, self.ask = {}, {}
        self.bid_n_shares, self.ask_n_shares = {}, {}
        self.last_prices = last_prices if last_prices is not None else [STARTING_PRICE]
        self.n_orders = 0

        # map of actor_id to actor
        self.actors = actors

    def quote_ask(self):
        min_price = 2147483647
        n_shares = None

        # shitty O(n) implementation
        for price in self.ask:
            if price < min_price:
                min_price = price
                n_shares = self.ask_n_shares[price]

        if not n_shares:
            return None
        else:
            return (min_price, n_shares)

    def quote_bid(self):
        max_price = -1
        n_shares = None

        # shitty O(n) implementation
        for price in self.bid:
            if price > max_price:
                max_price = price
                n_shares = self.bid_n_shares[price]

        if not n_shares:
            return None
        else:
            return (max_price, n_shares)

    def submit_buy(self, n_shares, price, actor_id):
        self.n_orders += 1

        if price in self.bid:
            self.bid[price].append((n_shares, actor_id, self.n_orders))
            self.bid_n_shares[price] += n_shares
        else:
            self.bid[price] = [(n_shares, actor_id, self.n_orders)]
            self.bid_n_shares[price] = n_shares

        return self.n_orders

    def submit_sell(self, n_shares, price, actor_id):
        self.n_orders += 1

        if price in self.ask:
            self.ask[price].append((n_shares, actor_id, self.n_orders))
            self.ask_n_shares[price] += n_shares
        else:
            self.ask[price] = [(n_shares, actor_id, self.n_orders)]
            self.ask_n_shares[price] = n_shares

        return self.n_orders

    def execute_step(self):
        for actor in self.actors.values():
            actor.execute_step()

        exec_by_price = []
        money_vol, exec_n_shares = 0, 0

        # try to execute as many orders as possible
        # accept overlapping orders not just exact matching
        # method 1: match one bid price to one ask price per step
        # TODO get this working
        bid_sorted = sorted([(k,v) for k,v in self.bid.items()], key=lambda x: x[0])
        ask_sorted = sorted([(k,v) for k,v in self.ask.items()], key=lambda x: x[0])

        ask_index = len(ask_sorted)-1
        matches = []

        for bid_index in range(len(bid_sorted)-2, -1, -1):
            while ask_sorted[ask_index][0] > bid_sorted[bid_index][0] and ask_index >= 0:
                ask_index -= 1
            if ask_index < 0:
                break
            matches.append((bid_index, ask_index))
            ask_index -= 1

        print(matches)

        for bid_index, ask_index in matches:
            bid_price, bid_orders = bid_sorted[bid_index]
            ask_price, ask_orders = ask_sorted[ask_index]

            vol = min([self.bid_n_shares[bid_price], self.ask_n_shares[ask_price]])

            assert bid_price >= ask_price

            exec_price = (bid_price + ask_price) / 2

            while len(bid_orders) > 0 and len(ask_orders) > 0:
                bid_n_shares, bid_actor_id, bid_order_id = bid_orders[-1]
                ask_n_shares, ask_actor_id, ask_order_id = ask_orders[-1]
                n_shares = min([bid_n_shares, ask_n_shares])

                self.actors[bid_actor_id].notify(bid_order_id, n_shares, exec_price)
                self.actors[ask_actor_id].notify(ask_order_id, n_shares, exec_price)

                exec_shares += n_shares

                if bid_n_shares > ask_n_shares:
                    ask_orders.pop()
                elif bid_n_shares < ask_n_shares:
                    bid_orders.pop()
                else:
                    bid_orders.pop()
                    ask_orders.pop()

            self.bid[bid_price] = bid_orders
            self.ask[ask_price] = ask_orders

            if len(bid_orders) == 0:
                del self.bid[bid_price]
                del self.bid_n_shares[bid_price]
            if len(ask_orders) == 0:
                del self.ask[ask_price]
                del self.ask_n_shares[ask_price]

            exec_by_price.append((exec_price, vol))
            money_vol += exec_price * vol
            exec_n_shares += vol

        # method 2: match by exact price
        """rm_bid_prices = []
        for price, bid_orders in self.bid.items():
            if price in self.ask:
                ask_orders = self.ask[price]
                vol = min([self.bid_n_shares[price], self.ask_n_shares[price]])
                exec_shares = 0

                self.bid_n_shares[price] -= vol
                self.ask_n_shares[price] -= vol

                while exec_shares < vol:
                    bid_n_shares, bid_actor_id, bid_order_id = bid_orders[-1]
                    ask_n_shares, ask_actor_id, ask_order_id = ask_orders[-1]
                    n_shares = min([bid_n_shares, ask_n_shares])

                    self.actors[bid_actor_id].notify(bid_order_id, n_shares)
                    self.actors[ask_actor_id].notify(ask_order_id, n_shares)

                    exec_shares += n_shares

                    if bid_n_shares > ask_n_shares:
                        ask_orders.pop()
                    elif bid_n_shares < ask_n_shares:
                        bid_orders.pop()
                    else:
                        bid_orders.pop()
                        ask_orders.pop()

                self.bid[price] = bid_orders
                self.ask[price] = ask_orders

                if len(bid_orders) == 0:
                    rm_bid_prices.append(price)
                if len(ask_orders) == 0:
                    del self.ask[price]
                    del self.ask_n_shares[price]

                exec_by_price.append((price, vol))
                money_vol += price * vol
                exec_n_shares += vol

        for price in rm_bid_prices:
            del self.bid[price]
            del self.bid_n_shares[price]"""

        # print current state
        print("execution volume by price: " + str(exec_by_price))
        print("bids: " + str(self.bid))
        print("asks: " + str(self.ask))

        if exec_n_shares > 0:
            self.last_prices.append(money_vol / exec_n_shares)
        else:
            self.last_prices.append(self.last_prices[-1])


def main():
    # TODO load real-world stock, pass to Stock() as last_prices
    # as a way of "seeding" the field
    prices_df = pd.read_csv("../WIKI-X-6mo-yfinance.csv")
    last_prices = list(prices_df["Close"].values[-WINDOW_SIZE:])

    #print(last_prices)

    sim = Stock(last_prices=last_prices)
    actors = {i:Actor(sim,i) for i in range(N_ACTORS)}
    sim.actors = actors

    for i in range(SIM_LENGTH):
        sim.execute_step()

    print(sim.last_prices)
    print([x.calculate_pnl() for x in actors.values()])

if __name__ == "__main__":
    main()