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

def get_data(symbol='ES=F'):
    """Fetch current market data for specified symbol"""
    try:
        data = yf.download(symbol, start='2020-01-01', group_by='ticker')
        
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

def create_chart(data, symbol, short_period=50, long_period=200, lookback_period=20):
    """Create interactive chart with price, moving averages, and breakout levels"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(f'{symbol} Price & Moving Averages', 'Volume', 'Breakout Levels'),
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
    
    # Breakout signals visualization
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
    
    # Strategy selection
    st.sidebar.header("📈 Strategy Selection")
    strategy_type = st.sidebar.selectbox(
        "Choose Strategy",
        ["SMA Crossover", "Breakout Strategy", "Both"],
        help="Select which trading strategy to analyze"
    )
    
    # Strategy parameters
    st.sidebar.header("⚙️ Strategy Parameters")
    
    # SMA parameters
    if strategy_type in ["SMA Crossover", "Both"]:
        st.sidebar.subheader("SMA Crossover")
        short_period = st.sidebar.slider("Short MA Period", 5, 100, 50)
        long_period = st.sidebar.slider("Long MA Period", 50, 300, 200)
    
    # Breakout parameters
    if strategy_type in ["Breakout Strategy", "Both"]:
        st.sidebar.subheader("Breakout Strategy")
        lookback_period = st.sidebar.slider("Lookback Period", 10, 50, 20)
        volume_factor = st.sidebar.slider("Volume Factor", 1.0, 3.0, 1.5, 0.1)
        stop_loss = st.sidebar.slider("Stop Loss %", 1, 20, 5) / 100
        take_profit = st.sidebar.slider("Take Profit %", 5, 50, 10) / 100
    
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
                data = get_data(symbol)
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
                short_period=short_period if strategy_type in ["SMA Crossover", "Both"] else 50,
                long_period=long_period if strategy_type in ["SMA Crossover", "Both"] else 200,
                lookback_period=lookback_period if strategy_type in ["Breakout Strategy", "Both"] else 20
            )
            st.plotly_chart(chart, use_container_width=True)
            
            # Run backtests
            st.subheader(f"📈 Strategy Results for {symbol}")
            
            backtest_cols = st.columns(2 if strategy_type == "Both" else 1)
            
            # SMA Crossover backtest
            if strategy_type in ["SMA Crossover", "Both"]:
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
            if strategy_type in ["Breakout Strategy", "Both"]:
                with backtest_cols[-1]:
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
                            st.success("� BREAKOUT - Price above resistance with volume")
                        elif current_price < current_low and current_volume > avg_volume * volume_factor:
                            st.error("� BREAKDOWN - Price below support with volume")
                        else:
                            st.info("⏳ CONSOLIDATION - Waiting for breakout")
            
            # Recent data table
            with st.expander(f"📋 Recent {symbol} Data"):
                st.dataframe(data.tail(10))
            
            st.markdown("---")
        
        # Market status
        st.header("🔴 Live Market Status")
        current_time = datetime.now()
        market_hours = "Market hours: 9:30 AM - 4:00 PM ET (Mon-Fri) | Futures: Nearly 24/5"
        st.info(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | {market_hours}")
        
        if st.session_state.last_update:
            st.success(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
    else:
        st.error("Failed to load market data. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
