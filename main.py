import yfinance as yf
import pandas as pd

# Download futures data with auto_adjust - no end date to get current data
data = yf.download('ES=F', start='2020-01-01', group_by='ticker')

# Flatten multi-level column
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(1)  # ['Close', 'High', 'Low', 'Open', 'Volume']

# Drop rows with any NaNs
data.dropna(inplace=True)

# Add OpenInterest column (Backtrader requires it)
data['OpenInterest'] = 0

# Capitalize column names to match Backtrader expectations
data.columns = [col.capitalize() for col in data.columns]

# Ensure datetime index
data.index = pd.to_datetime(data.index)

print(data.head())

import backtrader as bt

# Define a simple Moving Average Crossover strategy
class SMACross(bt.Strategy):
    params = (('short_period', 50), ('long_period', 200),)  # using 50-day and 200-day SMAs

    def __init__(self):
        # Initialize the two moving averages
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_period)
        self.sma_long  = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_period)

    def next(self):
        # Check if we are in the market
        if not self.position:  
            # If not in a position and short MA crosses above long MA, buy
            if self.sma_short[0] > self.sma_long[0]:
                self.buy()
        else:
            # If in a position and short MA falls below long MA, sell (exit)
            if self.sma_short[0] < self.sma_long[0]:
                self.sell()
# Prepare the data feed for Backtrader
data_feed = bt.feeds.PandasData(dataname=data)  # wrap the Pandas DataFrame

cerebro = bt.Cerebro()
cerebro.addstrategy(SMACross)          # add our strategy
cerebro.adddata(data_feed)             # add the data feed
cerebro.broker.set_cash(10000.0)       # starting capital ($10,000 for example)
cerebro.broker.setcommission(commission=0.0)  # (optional) set commission costs, e.g., 0 for demo

print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
results = cerebro.run()               # run the backtest
print(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")
