import streamlit as st
import yfinance as yf
import pandas as pd
import backtrader as bt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import threading
import queue
import numpy as np
import os
import requests
from urllib.parse import urlencode, urlparse, parse_qs
from dotenv import load_dotenv
import base64
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Strategy Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Market symbols and names
MARKETS = {
    'ES=F': 'E-mini S&P 500 Futures',
    'SPY': 'SPDR S&P 500 ETF',
    'QQQ': 'Invesco QQQ ETF',
    'BTC-USD': 'Bitcoin',
    'GLD': 'SPDR Gold Trust',
    'TLT': '20+ Year Treasury Bond ETF'
}

class SMACross(bt.Strategy):
    params = (('short_period', 50), ('long_period', 200),)

    def __init__(self):
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_period)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_period)
        self.trades = []
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        self.trades.append(f'{dt.isoformat()}: {txt}')

    def next(self):
        if not self.position:
            if self.sma_short[0] > self.sma_long[0]:
                self.log(f'BUY CREATE, Price: {self.data.close[0]:.2f}')
                self.buy()
        else:
            if self.sma_short[0] < self.sma_long[0]:
                self.log(f'SELL CREATE, Price: {self.data.close[0]:.2f}')
                self.sell()

class BreakoutStrategy(bt.Strategy):
    params = (
        ('lookback_period', 20),  # Period for high/low lookback
        ('volume_factor', 1.5),   # Volume must be X times average
        ('stop_loss', 0.05),      # 5% stop loss
        ('take_profit', 0.10),    # 10% take profit
    )

    def __init__(self):
        self.highest = bt.indicators.Highest(self.data.high, period=self.params.lookback_period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.params.lookback_period)
        self.volume_avg = bt.indicators.SimpleMovingAverage(self.data.volume, period=self.params.lookback_period)
        self.trades = []
        self.entry_price = None
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        self.trades.append(f'{dt.isoformat()}: {txt}')

    def next(self):
        if not self.position:
            # Check for breakout above resistance with volume confirmation
            if (self.data.close[0] > self.highest[-1] and 
                self.data.volume[0] > self.volume_avg[0] * self.params.volume_factor):
                self.log(f'BREAKOUT BUY, Price: {self.data.close[0]:.2f}, Volume: {self.data.volume[0]:,.0f}')
                self.entry_price = self.data.close[0]
                self.buy()
            
            # Check for breakdown below support with volume confirmation
            elif (self.data.close[0] < self.lowest[-1] and 
                  self.data.volume[0] > self.volume_avg[0] * self.params.volume_factor):
                self.log(f'BREAKDOWN SELL, Price: {self.data.close[0]:.2f}, Volume: {self.data.volume[0]:,.0f}')
                self.entry_price = self.data.close[0]
                self.sell()
        
        else:
            # Exit logic for long positions
            if self.position.size > 0 and self.entry_price:
                # Take profit
                if self.data.close[0] >= self.entry_price * (1 + self.params.take_profit):
                    self.log(f'TAKE PROFIT, Price: {self.data.close[0]:.2f}')
                    self.close()
                    self.entry_price = None
                # Stop loss
                elif self.data.close[0] <= self.entry_price * (1 - self.params.stop_loss):
                    self.log(f'STOP LOSS, Price: {self.data.close[0]:.2f}')
                    self.close()
                    self.entry_price = None
            
            # Exit logic for short positions
            elif self.position.size < 0 and self.entry_price:
                # Take profit
                if self.data.close[0] <= self.entry_price * (1 - self.params.take_profit):
                    self.log(f'TAKE PROFIT SHORT, Price: {self.data.close[0]:.2f}')
                    self.close()
                    self.entry_price = None
                # Stop loss
                elif self.data.close[0] >= self.entry_price * (1 + self.params.stop_loss):
                    self.log(f'STOP LOSS SHORT, Price: {self.data.close[0]:.2f}')
                    self.close()
                    self.entry_price = None

class MeanReversionStrategy(bt.Strategy):
    """
    MERN (Mean Reversion) Trading Algorithm
    
    Strategy Logic:
    1. Calculate the mean (SMA) and standard deviation of price over a lookback period
    2. Identify overbought/oversold conditions using Z-score (Bollinger Bands concept)
    3. Enter long when price is significantly below mean (oversold)
    4. Enter short when price is significantly above mean (overbought)
    5. Exit when price reverts back to mean or hits stop loss
    """
    params = (
        ('lookback_period', 20),    # Period for mean calculation
        ('z_score_threshold', 2.0), # Z-score threshold for entry signals
        ('z_score_exit', 0.5),      # Z-score threshold for mean reversion exit
        ('stop_loss', 0.03),        # 3% stop loss
        ('rsi_period', 14),         # RSI period for additional confirmation
        ('rsi_oversold', 30),       # RSI oversold level
        ('rsi_overbought', 70),     # RSI overbought level
        ('volume_confirmation', True), # Use volume confirmation
        ('min_volume_ratio', 1.2),  # Minimum volume ratio vs average
    )

    def __init__(self):
        # Mean and standard deviation calculation
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.lookback_period)
        self.std = bt.indicators.StandardDeviation(self.data.close, period=self.params.lookback_period)
        
        # Z-score calculation (how many standard deviations from mean)
        self.z_score = (self.data.close - self.sma) / self.std
        
        # RSI for additional momentum confirmation
        self.rsi = bt.indicators.RelativeStrengthIndex(period=self.params.rsi_period)
        
        # Volume analysis
        self.volume_sma = bt.indicators.SimpleMovingAverage(self.data.volume, period=self.params.lookback_period)
        
        # Bollinger Bands for visualization
        self.bb_upper = self.sma + (self.std * self.params.z_score_threshold)
        self.bb_lower = self.sma - (self.std * self.params.z_score_threshold)
        
        # Trade tracking
        self.trades = []
        self.entry_price = None
        self.entry_z_score = None
        
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        self.trades.append(f'{dt.isoformat()}: {txt}')

    def next(self):
        current_price = self.data.close[0]
        current_z_score = self.z_score[0]
        current_rsi = self.rsi[0]
        current_volume = self.data.volume[0]
        avg_volume = self.volume_sma[0]
        
        # Volume confirmation check
        volume_ok = True
        if self.params.volume_confirmation:
            volume_ok = current_volume >= avg_volume * self.params.min_volume_ratio
        
        if not self.position:
            # LONG ENTRY: Price significantly below mean (oversold)
            if (current_z_score <= -self.params.z_score_threshold and 
                current_rsi <= self.params.rsi_oversold and
                volume_ok):
                
                self.log(f'MEAN REVERSION LONG ENTRY - Price: {current_price:.2f}, '
                        f'Z-Score: {current_z_score:.2f}, RSI: {current_rsi:.1f}, '
                        f'Volume Ratio: {current_volume/avg_volume:.2f}')
                self.entry_price = current_price
                self.entry_z_score = current_z_score
                self.buy()
            
            # SHORT ENTRY: Price significantly above mean (overbought)
            elif (current_z_score >= self.params.z_score_threshold and 
                  current_rsi >= self.params.rsi_overbought and
                  volume_ok):
                
                self.log(f'MEAN REVERSION SHORT ENTRY - Price: {current_price:.2f}, '
                        f'Z-Score: {current_z_score:.2f}, RSI: {current_rsi:.1f}, '
                        f'Volume Ratio: {current_volume/avg_volume:.2f}')
                self.entry_price = current_price
                self.entry_z_score = current_z_score
                self.sell()
        
        else:
            # EXIT LOGIC
            
            # Long position exits
            if self.position.size > 0:
                # Mean reversion exit: price has reverted back towards mean
                if current_z_score >= -self.params.z_score_exit:
                    self.log(f'MEAN REVERSION EXIT LONG - Price: {current_price:.2f}, '
                            f'Z-Score: {current_z_score:.2f}')
                    self.close()
                    self.entry_price = None
                    self.entry_z_score = None
                
                # Stop loss
                elif current_price <= self.entry_price * (1 - self.params.stop_loss):
                    self.log(f'STOP LOSS LONG - Entry: {self.entry_price:.2f}, '
                            f'Exit: {current_price:.2f}')
                    self.close()
                    self.entry_price = None
                    self.entry_z_score = None
                
                # RSI momentum reversal (price moving against us)
                elif current_rsi >= self.params.rsi_overbought:
                    self.log(f'RSI REVERSAL EXIT LONG - Price: {current_price:.2f}, '
                            f'RSI: {current_rsi:.1f}')
                    self.close()
                    self.entry_price = None
                    self.entry_z_score = None
            
            # Short position exits
            elif self.position.size < 0:
                # Mean reversion exit: price has reverted back towards mean
                if current_z_score <= self.params.z_score_exit:
                    self.log(f'MEAN REVERSION EXIT SHORT - Price: {current_price:.2f}, '
                            f'Z-Score: {current_z_score:.2f}')
                    self.close()
                    self.entry_price = None
                    self.entry_z_score = None
                
                # Stop loss
                elif current_price >= self.entry_price * (1 + self.params.stop_loss):
                    self.log(f'STOP LOSS SHORT - Entry: {self.entry_price:.2f}, '
                            f'Exit: {current_price:.2f}')
                    self.close()
                    self.entry_price = None
                    self.entry_z_score = None
                
                # RSI momentum reversal (price moving against us)
                elif current_rsi <= self.params.rsi_oversold:
                    self.log(f'RSI REVERSAL EXIT SHORT - Price: {current_price:.2f}, '
                            f'RSI: {current_rsi:.1f}')
                    self.close()
                    self.entry_price = None
                    self.entry_z_score = None

def get_data(symbol='ES=F', period='5d', interval='5m'):
    """Fetch current market data for specified symbol with intraday intervals"""
    try:
        # For real-time trading, use shorter periods with minute intervals
        data = yf.download(symbol, period=period, interval=interval, group_by='ticker')
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(1)
        
        data.dropna(inplace=True)
        if 'Openinterest' not in data.columns:
            data['Openinterest'] = 0
        data.columns = [col.capitalize() for col in data.columns]
        data.index = pd.to_datetime(data.index)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def run_backtest(data, strategy_class=SMACross, **kwargs):
    """Run backtest and return results"""
    try:
        data_feed = bt.feeds.PandasData(dataname=data)
        
        cerebro = bt.Cerebro()
        strategy = cerebro.addstrategy(strategy_class, **kwargs)
        cerebro.adddata(data_feed)
        cerebro.broker.set_cash(10000.0)
        cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
        
        starting_value = cerebro.broker.getvalue()
        results = cerebro.run()
        final_value = cerebro.broker.getvalue()
        
        return {
            'starting_value': starting_value,
            'final_value': final_value,
            'return_pct': ((final_value - starting_value) / starting_value) * 100,
            'strategy': results[0],
            'trades': results[0].trades if hasattr(results[0], 'trades') else []
        }
    except Exception as e:
        st.error(f"Error running backtest: {e}")
        return None

def create_chart(data, symbol, short_period=50, long_period=200, lookback_period=20, strategy_type="Both"):
    """Create interactive chart with price, moving averages, breakout levels, and mean reversion indicators"""
    if strategy_type == "Mean Reversion":
        subplot_titles = (f'{symbol} Price & Mean Reversion', 'Volume', 'RSI & Z-Score')
    else:
        subplot_titles = (f'{symbol} Price & Moving Averages', 'Volume', 'Indicators')
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    if strategy_type in ["SMA Crossover", "Both"]:
        # Calculate and plot moving averages
        sma_short = data['Close'].rolling(window=short_period).mean()
        sma_long = data['Close'].rolling(window=long_period).mean()
        
        fig.add_trace(
            go.Scatter(x=data.index, y=sma_short, name=f'SMA {short_period}', line=dict(color='orange')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data.index, y=sma_long, name=f'SMA {long_period}', line=dict(color='red')),
            row=1, col=1
        )
    
    if strategy_type in ["Mean Reversion", "Both"]:
        # Calculate mean reversion indicators
        sma_mean = data['Close'].rolling(window=lookback_period).mean()
        std_dev = data['Close'].rolling(window=lookback_period).std()
        
        # Bollinger Bands (2 standard deviations)
        bb_upper = sma_mean + (std_dev * 2)
        bb_lower = sma_mean - (std_dev * 2)
        
        # Plot mean and Bollinger Bands
        fig.add_trace(
            go.Scatter(x=data.index, y=sma_mean, name=f'Mean ({lookback_period})', 
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data.index, y=bb_upper, name='Upper Band (2σ)', 
                      line=dict(color='red', dash='dash'), fill=None),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data.index, y=bb_lower, name='Lower Band (2σ)', 
                      line=dict(color='green', dash='dash'), 
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    if strategy_type in ["Breakout Strategy", "Both"] and strategy_type != "Mean Reversion":
        # Calculate breakout levels
        highest = data['High'].rolling(window=lookback_period).max()
        lowest = data['Low'].rolling(window=lookback_period).min()
        
        fig.add_trace(
            go.Scatter(x=data.index, y=highest, name=f'Resistance ({lookback_period}d)', 
                      line=dict(color='green', dash='dash')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data.index, y=lowest, name=f'Support ({lookback_period}d)', 
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
    
    # Volume chart
    colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # Volume moving average
    volume_avg = data['Volume'].rolling(window=lookback_period).mean()
    fig.add_trace(
        go.Scatter(x=data.index, y=volume_avg, name=f'Volume MA {lookback_period}', 
                  line=dict(color='blue')),
        row=2, col=1
    )
    
    # Third subplot: Strategy-specific indicators
    if strategy_type == "Mean Reversion":
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Z-Score
        sma_mean = data['Close'].rolling(window=lookback_period).mean()
        std_dev = data['Close'].rolling(window=lookback_period).std()
        z_score = (data['Close'] - sma_mean) / std_dev
        
        # Plot RSI
        fig.add_trace(
            go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        
        # Update y-axis for RSI subplot
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        
    elif strategy_type in ["Breakout Strategy", "Both"]:
        # Breakout signals visualization
        highest = data['High'].rolling(window=lookback_period).max()
        lowest = data['Low'].rolling(window=lookback_period).min()
        breakout_up = data['Close'] > highest.shift(1)
        breakout_down = data['Close'] < lowest.shift(1)
        
        fig.add_trace(
            go.Scatter(x=data.index, y=breakout_up.astype(int), name='Breakout Up', 
                      line=dict(color='green'), fill='tozeroy'),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data.index, y=breakout_down.astype(int) * -1, name='Breakdown', 
                      line=dict(color='red'), fill='tozeroy'),
            row=3, col=1
        )
    
    fig.update_layout(
        title=f'{symbol} Multi-Strategy Analysis',
        xaxis_rangeslider_visible=False,
        height=900
    )
    
    return fig

# Utility to get Schwab credentials from environment

# Try to get from Streamlit secrets first (for deployment), then from environment variables (for local development)
try:
    SCHWAB_CLIENT_ID = st.secrets.get('SCHWAB_CLIENT_ID') or os.environ.get('SCHWAB_CLIENT_ID')
    SCHWAB_CLIENT_SECRET = st.secrets.get('SCHWAB_CLIENT_SECRET') or os.environ.get('SCHWAB_CLIENT_SECRET')
    SCHWAB_REDIRECT_URI = st.secrets.get('SCHWAB_REDIRECT_URI') or os.environ.get('SCHWAB_REDIRECT_URI', 'https://127.0.0.1')
except:
    # Fall back to environment variables if secrets are not available
    SCHWAB_CLIENT_ID = os.environ.get('SCHWAB_CLIENT_ID')
    SCHWAB_CLIENT_SECRET = os.environ.get('SCHWAB_CLIENT_SECRET')
    SCHWAB_REDIRECT_URI = os.environ.get('SCHWAB_REDIRECT_URI', 'https://127.0.0.1')

SCHWAB_AUTH_URL = "https://api.schwabapi.com/v1/oauth/authorize"
SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

def oauth2(client_id, client_secret, redirect_uri, auth_url, token_url, scope="read", state="random_state_string"): 
    """
    Schwab OAuth2 utility function. Handles:
    - Generating the authorization URL
    - Exchanging code for access/refresh tokens
    - Refreshing access token if expired
    Returns: (access_token, refresh_token, login_status, auth_url)
    """
    import requests
    from urllib.parse import urlencode
    import streamlit as st

    # Session state keys
    if 'schwab_access_token' not in st.session_state:
        st.session_state.schwab_access_token = None
    if 'schwab_refresh_token' not in st.session_state:
        st.session_state.schwab_refresh_token = None
    if 'schwab_code' not in st.session_state:
        st.session_state.schwab_code = None
    if 'schwab_token_expires_at' not in st.session_state:
        st.session_state.schwab_token_expires_at = None

    # Step 1: Build authorization URL
    params = {
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': scope,
        'state': state
    }
    auth_url_full = f"{auth_url}?{urlencode(params)}"

    # Step 2: Handle callback with code
    query_params = st.query_params
    if 'code' in query_params and not st.session_state.schwab_access_token:
        code = query_params['code'][0] if isinstance(query_params['code'], list) else query_params['code']
        st.session_state.schwab_code = code
        # Exchange code for tokens
        credentials = f"{client_id}:{client_secret}"
        base64_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        
        token_data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': redirect_uri,
        }
        headers = {
            "Authorization": f"Basic {base64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = requests.post(token_url, headers=headers, data=token_data)
        if response.status_code == 200:
            token_json = response.json()
            st.session_state.schwab_access_token = token_json.get('access_token')
            st.session_state.schwab_refresh_token = token_json.get('refresh_token')
            expires_in = token_json.get('expires_in', 1800)  # seconds
            st.session_state.schwab_token_expires_at = datetime.utcnow().timestamp() + int(expires_in)
            login_status = True
        else:
            st.error(f"Token exchange failed: {response.text}")
            login_status = False
    # Step 3: Refresh token if expired
    elif st.session_state.schwab_refresh_token and not st.session_state.schwab_access_token:
        credentials = f"{client_id}:{client_secret}"
        base64_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        
        refresh_data = {
            'grant_type': 'refresh_token',
            'refresh_token': st.session_state.schwab_refresh_token,
        }
        headers = {
            "Authorization": f"Basic {base64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = requests.post(token_url, headers=headers, data=refresh_data)
        if response.status_code == 200:
            token_json = response.json()
            st.session_state.schwab_access_token = token_json.get('access_token')
            st.session_state.schwab_refresh_token = token_json.get('refresh_token')
            expires_in = token_json.get('expires_in', 1800)
            st.session_state.schwab_token_expires_at = datetime.utcnow().timestamp() + int(expires_in)
            login_status = True
        else:
            st.error(f"Token refresh failed: {response.text}")
            login_status = False
    else:
        login_status = st.session_state.schwab_access_token is not None

    return (
        st.session_state.schwab_access_token,
        st.session_state.schwab_refresh_token,
        login_status,
        auth_url_full
    )

def get_schwab_access_token():
    """
    Retrieve a valid Schwab access token from session state, refreshing if needed.
    Returns None if not authenticated.
    """
    # Check if access token is present and not expired
    access_token = st.session_state.get('schwab_access_token')
    expires_at = st.session_state.get('schwab_token_expires_at')
    refresh_token = st.session_state.get('schwab_refresh_token')
    now = datetime.utcnow().timestamp()

    # If access token exists and not expired, return it
    if access_token and expires_at and now < expires_at - 60:
        return access_token

    # If refresh token exists, try to refresh
    if refresh_token:
        st.info("Refreshing Schwab access token...")
        credentials = f"{SCHWAB_CLIENT_ID}:{SCHWAB_CLIENT_SECRET}"
        base64_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        
        token_data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
        }
        headers = {
            "Authorization": f"Basic {base64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            response = requests.post(SCHWAB_TOKEN_URL, headers=headers, data=token_data)
            if response.status_code == 200:
                token_json = response.json()
                st.session_state.schwab_access_token = token_json.get('access_token')
                st.session_state.schwab_refresh_token = token_json.get('refresh_token', refresh_token)
                # Schwab returns expires_in in seconds
                expires_in = token_json.get('expires_in', 1800)  # default 30min
                st.session_state.schwab_token_expires_at = now + int(expires_in)
                return st.session_state.schwab_access_token
            else:
                st.error(f"Failed to refresh Schwab token: {response.text}")
                st.session_state.schwab_access_token = None
                st.session_state.schwab_refresh_token = None
                st.session_state.schwab_token_expires_at = None
                return None
        except Exception as e:
            st.error(f"Exception during Schwab token refresh: {e}")
            return None
    return None

def get_schwab_quotes(symbols):
    """Fetch real-time quotes for given symbols from Schwab API."""
    access_token = get_schwab_access_token()
    if not access_token:
        st.warning("Schwab API access token not available. Please login via OAuth2.")
        return None
    # Schwab API endpoint for quotes
    url = f"https://api.schwabapi.com/marketdata/v1/quotes?symbols={','.join(symbols)}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            # Token might be expired, try to refresh and retry once
            st.info("Access token expired, attempting to refresh...")
            access_token = get_schwab_access_token()  # This will refresh if possible
            if not access_token:
                st.error("Failed to refresh Schwab access token. Please login again.")
                return None
            headers["Authorization"] = f"Bearer {access_token}"
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Schwab API error after refresh: {response.text}")
                return None
        else:
            st.error(f"Schwab API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception during Schwab API call: {e}")
        return None

def schwab_oauth_flow():
    st.subheader("Schwab OAuth2 Login")
    if 'schwab_access_token' not in st.session_state:
        st.session_state.schwab_access_token = None
    if 'schwab_refresh_token' not in st.session_state:
        st.session_state.schwab_refresh_token = None
    if 'schwab_code' not in st.session_state:
        st.session_state.schwab_code = None
    if 'schwab_token_expires_at' not in st.session_state:
        st.session_state.schwab_token_expires_at = None

    # Step 1: Direct user to Schwab's authorization URL
    params = {
        'response_type': 'code',
        'client_id': SCHWAB_CLIENT_ID,
        'redirect_uri': SCHWAB_REDIRECT_URI,
        'scope': 'read',  # Adjust scope as needed
        'state': 'random_state_string'  # Optional, for CSRF protection
    }
    auth_url = f"{SCHWAB_AUTH_URL}?{urlencode(params)}"
    st.markdown(f"[Login with Schwab]({auth_url})", unsafe_allow_html=True)

    # Manual code input for when redirect doesn't work
    st.subheader("Manual Authorization Code Input")
    st.info("If the redirect doesn't work, copy the authorization code from the URL and paste it here:")
    manual_code = st.text_input("Authorization Code", placeholder="Paste the code from the URL here...")
    
    if manual_code and not st.session_state.schwab_access_token:
        st.success("Authorization code received. Exchanging for access token...")
        # Exchange code for access token
        credentials = f"{SCHWAB_CLIENT_ID}:{SCHWAB_CLIENT_SECRET}"
        base64_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        
        token_data = {
            'grant_type': 'authorization_code',
            'code': manual_code,
            'redirect_uri': SCHWAB_REDIRECT_URI,
        }
        headers = {
            "Authorization": f"Basic {base64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            response = requests.post(SCHWAB_TOKEN_URL, headers=headers, data=token_data)
            if response.status_code == 200:
                token_json = response.json()
                st.session_state.schwab_access_token = token_json.get('access_token')
                st.session_state.schwab_refresh_token = token_json.get('refresh_token')
                expires_in = token_json.get('expires_in', 1800)  # seconds
                st.session_state.schwab_token_expires_at = datetime.utcnow().timestamp() + int(expires_in)
                st.success("Schwab access token obtained!")
            else:
                st.error(f"Token exchange failed: {response.text}")
        except Exception as e:
            st.error(f"Exception during token exchange: {e}")

    # Step 2: User is redirected back with ?code=...
    query_params = st.query_params
    if 'code' in query_params and not st.session_state.schwab_access_token:
        code = query_params['code'][0] if isinstance(query_params['code'], list) else query_params['code']
        st.session_state.schwab_code = code
        st.success("Authorization code received. Exchanging for access token...")
        # Step 3: Exchange code for access token
        credentials = f"{SCHWAB_CLIENT_ID}:{SCHWAB_CLIENT_SECRET}"
        base64_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        
        token_data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': SCHWAB_REDIRECT_URI,
        }
        headers = {
            "Authorization": f"Basic {base64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            response = requests.post(SCHWAB_TOKEN_URL, headers=headers, data=token_data)
            if response.status_code == 200:
                token_json = response.json()
                st.session_state.schwab_access_token = token_json.get('access_token')
                st.session_state.schwab_refresh_token = token_json.get('refresh_token')
                expires_in = token_json.get('expires_in', 1800)  # seconds
                st.session_state.schwab_token_expires_at = datetime.utcnow().timestamp() + int(expires_in)
                st.success("Schwab access token obtained!")
            else:
                st.error(f"Token exchange failed: {response.text}")
        except Exception as e:
            st.error(f"Exception during token exchange: {e}")

    # Show current token status
    if st.session_state.schwab_access_token:
        expires_at = st.session_state.get('schwab_token_expires_at')
        if expires_at:
            expires_in = int(expires_at - datetime.utcnow().timestamp())
            st.info(f"Schwab API is authenticated. Token expires in {expires_in//60}m {expires_in%60}s.")
        else:
            st.info("Schwab API is authenticated.")
        if st.button("Logout Schwab API"):
            st.session_state.schwab_access_token = None
            st.session_state.schwab_refresh_token = None
            st.session_state.schwab_token_expires_at = None
            st.session_state.schwab_code = None
            st.success("Logged out from Schwab API.")
    else:
        st.info("Not authenticated with Schwab API yet.")

# Main app
def main():
    st.title("📈 Multi-Strategy Trading Dashboard")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    refresh_button = st.sidebar.button("🔄 Refresh Data Now")
    
    # Market selection
    st.sidebar.header("📊 Market Selection")
    selected_markets = st.sidebar.multiselect(
        "Choose Markets",
        options=list(MARKETS.keys()),
        default=['ES=F', 'SPY'],
        help="Select up to 4 markets for analysis"
    )
    
    # Data interval selection
    st.sidebar.header("⏰ Data Frequency")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        period = st.sidebar.selectbox(
            "Time Period",
            options=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y'],
            index=1,  # Default to 5d
            help="Historical data period to fetch"
        )
    
    with col2:
        interval = st.sidebar.selectbox(
            "Data Interval",
            options=['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d'],
            index=2,  # Default to 5m
            help="Data granularity (1m-90m only work with periods ≤60 days)"
        )
    
    # Display interval warning
    if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m'] and period in ['6mo', '1y', '2y']:
        st.sidebar.warning("⚠️ Minute intervals only work with periods ≤60 days. Using 1h interval instead.")
        interval = '1h'
    
    # Strategy selection
    st.sidebar.header("📈 Strategy Selection")
    strategy_type = st.sidebar.selectbox(
        "Choose Strategy",
        ["SMA Crossover", "Breakout Strategy", "Mean Reversion", "All Strategies"],
        help="Select which trading strategy to analyze"
    )
    
    # Strategy parameters
    st.sidebar.header("⚙️ Strategy Parameters")
    
    # Adjust default parameters based on interval
    if interval in ['1m', '2m', '5m']:
        default_short = 20  # 20 periods for 5min = 100 minutes
        default_long = 50   # 50 periods for 5min = 250 minutes
        default_lookback = 10  # 10 periods for breakout
        max_period = 200
    elif interval in ['15m', '30m']:
        default_short = 12  # 12 periods for 15min = 3 hours
        default_long = 24   # 24 periods for 15min = 6 hours
        default_lookback = 8
        max_period = 100
    else:  # 1h, 1d
        default_short = 50
        default_long = 200
        default_lookback = 20
        max_period = 300
    
    # SMA parameters
    if strategy_type in ["SMA Crossover", "All Strategies"]:
        st.sidebar.subheader("SMA Crossover")
        short_period = st.sidebar.slider("Short MA Period", 5, max_period, default_short,
                                        help=f"Periods for short MA ({interval} intervals)")
        long_period = st.sidebar.slider("Long MA Period", 10, max_period, default_long,
                                       help=f"Periods for long MA ({interval} intervals)")
    else:
        short_period = default_short
        long_period = default_long
    
    # Breakout parameters
    if strategy_type in ["Breakout Strategy", "All Strategies"]:
        st.sidebar.subheader("Breakout Strategy")
        lookback_period = st.sidebar.slider("Lookback Period", 5, 100, default_lookback,
                                          help=f"Periods for support/resistance ({interval} intervals)")
        volume_factor = st.sidebar.slider("Volume Factor", 1.0, 3.0, 1.5, 0.1)
        stop_loss = st.sidebar.slider("Stop Loss %", 1, 20, 5) / 100
        take_profit = st.sidebar.slider("Take Profit %", 5, 50, 10) / 100
    else:
        lookback_period = default_lookback
        volume_factor = 1.5
        stop_loss = 0.05
        take_profit = 0.10
    
    # Mean Reversion parameters
    if strategy_type in ["Mean Reversion", "All Strategies"]:
        st.sidebar.subheader("Mean Reversion (MERN)")
        mean_lookback = st.sidebar.slider("Mean Lookback Period", 10, 100, default_lookback,
                                        help=f"Periods for mean calculation ({interval} intervals)")
        z_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 3.0, 2.0, 0.1,
                                      help="Standard deviations from mean for entry")
        z_exit = st.sidebar.slider("Z-Score Exit", 0.1, 1.0, 0.5, 0.1,
                                 help="Z-score level for mean reversion exit")
        mean_stop_loss = st.sidebar.slider("Mean Rev Stop Loss %", 1, 10, 3) / 100
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
        rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 40, 30)
        rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 90, 70)
        volume_confirmation = st.sidebar.checkbox("Volume Confirmation", value=True)
        min_volume_ratio = st.sidebar.slider("Min Volume Ratio", 1.0, 3.0, 1.2, 0.1)
    else:
        mean_lookback = default_lookback
        z_threshold = 2.0
        z_exit = 0.5
        mean_stop_loss = 0.03
        rsi_period = 14
        rsi_oversold = 30
        rsi_overbought = 70
        volume_confirmation = True
        min_volume_ratio = 1.2
    
    # Initialize session state
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'market_data' not in st.session_state:
        st.session_state.market_data = {}
    
    # Auto refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Fetch data for selected markets
    if refresh_button or not st.session_state.market_data or auto_refresh:
        with st.spinner("Fetching latest market data..."):
            st.session_state.market_data = {}
            for symbol in selected_markets:
                data = get_data(symbol, period=period, interval=interval)
                if data is not None:
                    st.session_state.market_data[symbol] = data
            st.session_state.last_update = datetime.now()
    
    if st.session_state.market_data:
        # Display market overview
        st.header("🌍 Market Overview")
        cols = st.columns(len(selected_markets))
        
        for i, (symbol, data) in enumerate(st.session_state.market_data.items()):
            with cols[i]:
                current_price = data['Close'].iloc[-1]
                daily_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                daily_change_pct = (daily_change / data['Close'].iloc[-2]) * 100
                
                st.metric(
                    f"{symbol}",
                    f"${current_price:.2f}",
                    f"{daily_change_pct:.2f}%"
                )
        
        # Display charts for each market
        for symbol, data in st.session_state.market_data.items():
            st.header(f"📊 {symbol} - {MARKETS[symbol]}")
            
            # Create and display chart
            chart = create_chart(
                data, symbol, 
                short_period=short_period if strategy_type in ["SMA Crossover", "All Strategies"] else 50,
                long_period=long_period if strategy_type in ["SMA Crossover", "All Strategies"] else 200,
                lookback_period=mean_lookback if strategy_type == "Mean Reversion" else (lookback_period if strategy_type in ["Breakout Strategy", "All Strategies"] else 20),
                strategy_type=strategy_type
            )
            st.plotly_chart(chart, use_container_width=True)
            
            # Run backtests
            st.subheader(f"📈 Strategy Results for {symbol}")
            
            if strategy_type == "All Strategies":
                backtest_cols = st.columns(3)
            elif strategy_type in ["SMA Crossover", "Breakout Strategy", "Mean Reversion"]:
                backtest_cols = st.columns(1)
            else:
                backtest_cols = st.columns(2)
            
            # SMA Crossover backtest
            if strategy_type in ["SMA Crossover", "All Strategies"]:
                with backtest_cols[0]:
                    st.markdown("**SMA Crossover Strategy**")
                    sma_results = run_backtest(
                        data, SMACross, 
                        short_period=short_period, 
                        long_period=long_period
                    )
                    
                    if sma_results:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Starting", f"${sma_results['starting_value']:,.0f}")
                        with col2:
                            st.metric("Final", f"${sma_results['final_value']:,.0f}")
                        with col3:
                            st.metric("Return", f"{sma_results['return_pct']:.1f}%")
                        
                        # Current SMA signal
                        sma_short = data['Close'].rolling(window=short_period).mean().iloc[-1]
                        sma_long_val = data['Close'].rolling(window=long_period).mean().iloc[-1]
                        
                        if sma_short > sma_long_val:
                            st.success("🟢 BULLISH - Short MA above Long MA")
                        else:
                            st.error("🔴 BEARISH - Short MA below Long MA")
            
            # Breakout strategy backtest
            if strategy_type in ["Breakout Strategy", "All Strategies"]:
                col_idx = 1 if strategy_type == "All Strategies" else 0
                with backtest_cols[col_idx]:
                    st.markdown("**Breakout Strategy**")
                    breakout_results = run_backtest(
                        data, BreakoutStrategy,
                        lookback_period=lookback_period,
                        volume_factor=volume_factor,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    if breakout_results:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Starting", f"${breakout_results['starting_value']:,.0f}")
                        with col2:
                            st.metric("Final", f"${breakout_results['final_value']:,.0f}")
                        with col3:
                            st.metric("Return", f"{breakout_results['return_pct']:.1f}%")
                        
                        # Current breakout signal
                        current_high = data['High'].rolling(window=lookback_period).max().iloc[-1]
                        current_low = data['Low'].rolling(window=lookback_period).min().iloc[-1]
                        current_price = data['Close'].iloc[-1]
                        current_volume = data['Volume'].iloc[-1]
                        avg_volume = data['Volume'].rolling(window=lookback_period).mean().iloc[-1]
                        
                        if current_price > current_high and current_volume > avg_volume * volume_factor:
                            st.success("🟢 BREAKOUT - Price above resistance with volume")
                        elif current_price < current_low and current_volume > avg_volume * volume_factor:
                            st.error("🔴 BREAKDOWN - Price below support with volume")
                        else:
                            st.info("⏳ CONSOLIDATION - Waiting for breakout")
            
            # Mean Reversion strategy backtest
            if strategy_type in ["Mean Reversion", "All Strategies"]:
                col_idx = 2 if strategy_type == "All Strategies" else 0
                with backtest_cols[col_idx]:
                    st.markdown("**Mean Reversion (MERN) Strategy**")
                    mean_rev_results = run_backtest(
                        data, MeanReversionStrategy,
                        lookback_period=mean_lookback,
                        z_score_threshold=z_threshold,
                        z_score_exit=z_exit,
                        stop_loss=mean_stop_loss,
                        rsi_period=rsi_period,
                        rsi_oversold=rsi_oversold,
                        rsi_overbought=rsi_overbought,
                        volume_confirmation=volume_confirmation,
                        min_volume_ratio=min_volume_ratio
                    )
                    
                    if mean_rev_results:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Starting", f"${mean_rev_results['starting_value']:,.0f}")
                        with col2:
                            st.metric("Final", f"${mean_rev_results['final_value']:,.0f}")
                        with col3:
                            st.metric("Return", f"{mean_rev_results['return_pct']:.1f}%")
                        
                        # Current mean reversion signal
                        current_price = data['Close'].iloc[-1]
                        sma_mean = data['Close'].rolling(window=mean_lookback).mean().iloc[-1]
                        std_dev = data['Close'].rolling(window=mean_lookback).std().iloc[-1]
                        current_z_score = (current_price - sma_mean) / std_dev
                        
                        # Calculate RSI
                        delta = data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean().iloc[-1]
                        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean().iloc[-1]
                        current_rsi = 100 - (100 / (1 + gain / loss)) if loss != 0 else 50
                        
                        st.metric("Z-Score", f"{current_z_score:.2f}", help="Standard deviations from mean")
                        st.metric("RSI", f"{current_rsi:.1f}", help="Relative Strength Index")
                        
                        if current_z_score <= -z_threshold and current_rsi <= rsi_oversold:
                            st.success("🟢 OVERSOLD - Mean reversion buy signal")
                        elif current_z_score >= z_threshold and current_rsi >= rsi_overbought:
                            st.error("🔴 OVERBOUGHT - Mean reversion sell signal")
                        elif abs(current_z_score) <= z_exit:
                            st.info("🎯 FAIR VALUE - Price near mean")
                        else:
                            st.warning("⏳ TRENDING - Price deviating from mean")
            
            # Recent data table
            with st.expander(f"📋 Recent {symbol} Data ({interval} intervals)"):
                recent_data = data.tail(10).copy()
                # Format timestamp to show time
                recent_data.index = recent_data.index.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(recent_data)
            
            st.markdown("---")
        
        # Market status
        st.header("🔴 Live Market Status")
        current_time = datetime.now()
        
        # Format interval display
        interval_display = {
            '1m': '1 minute', '2m': '2 minutes', '5m': '5 minutes',
            '15m': '15 minutes', '30m': '30 minutes', '60m': '1 hour',
            '90m': '90 minutes', '1h': '1 hour', '1d': '1 day'
        }
        
        market_hours = "Market hours: 9:30 AM - 4:00 PM ET (Mon-Fri) | Futures: Nearly 24/5"
        data_info = f"Data Frequency: {interval_display.get(interval, interval)} | Period: {period}"
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | {market_hours}")
        with col2:
            st.info(f"{data_info}")
        
        if st.session_state.last_update:
            st.success(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')} | Next refresh in {30 - (current_time.second % 30)} seconds" if auto_refresh else f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        # Schwab API section
        st.header("🔗 Schwab API: Defense Stocks (PLTR, LMT, BA)")
        schwab_symbols = ["PLTR", "LMT", "BA"]
        schwab_data = get_schwab_quotes(schwab_symbols)
        if schwab_data:
            st.write(schwab_data)
        else:
            st.info("Schwab API integration placeholder. Implement OAuth2 and endpoint calls to display real data.")
        
        # Schwab OAuth2 login section
        st.header("🔑 Schwab API Login")
        schwab_oauth_flow()
    else:
        st.error("Failed to load market data. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
