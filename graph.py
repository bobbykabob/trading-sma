import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download futures data
data = yf.download('ES=F', start='2020-01-01', end='2023-12-31', group_by='ticker')

# Flatten multi-index
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(1)

data.dropna(inplace=True)
data.columns = [col.capitalize() for col in data.columns]
data.index = pd.to_datetime(data.index)

# Compute SMAs
data['Sma50'] = data['Close'].rolling(window=50).mean()
data['Sma200'] = data['Close'].rolling(window=200).mean()

# Plot
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close', alpha=0.8)
plt.plot(data.index, data['Sma50'], label='50-Day SMA')
plt.plot(data.index, data['Sma200'], label='200-Day SMA')
plt.title('ES=F Close Price with 50 and 200-Day SMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
