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

# Configure Streamlit page
st.set_page_config(
    page_title="Real-Time Trading Strategy",
    page_icon="📈",
    layout="wide"
)

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

def get_data():
    """Fetch current market data"""
    try:
        data = yf.download('ES=F', start='2020-01-01', group_by='ticker')
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(1)
        
        data.dropna(inplace=True)
        data['OpenInterest'] = 0
        data.columns = [col.capitalize() for col in data.columns]
        data.index = pd.to_datetime(data.index)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def run_backtest(data):
    """Run backtest and return results"""
    try:
        data_feed = bt.feeds.PandasData(dataname=data)
        
        cerebro = bt.Cerebro()
        strategy = cerebro.addstrategy(SMACross)
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

def create_chart(data):
    """Create interactive chart with price and moving averages"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('ES=F Price & Moving Averages', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='ES=F'
        ),
        row=1, col=1
    )
    
    # Calculate and plot moving averages
    sma_50 = data['Close'].rolling(window=50).mean()
    sma_200 = data['Close'].rolling(window=200).mean()
    
    fig.add_trace(
        go.Scatter(x=data.index, y=sma_50, name='SMA 50', line=dict(color='orange')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=sma_200, name='SMA 200', line=dict(color='red')),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    fig.update_layout(
        title='ES=F Real-Time Trading Analysis',
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    return fig

# Main app
def main():
    st.title("📈 Real-Time Trading Strategy Dashboard")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    refresh_button = st.sidebar.button("Refresh Data Now")
    
    # Strategy parameters
    st.sidebar.header("Strategy Parameters")
    short_period = st.sidebar.slider("Short MA Period", 5, 100, 50)
    long_period = st.sidebar.slider("Long MA Period", 50, 300, 200)
    
    # Initialize session state
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Auto refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Fetch data
    if refresh_button or st.session_state.data is None or auto_refresh:
        with st.spinner("Fetching latest market data..."):
            st.session_state.data = get_data()
            st.session_state.last_update = datetime.now()
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
        
        with col2:
            daily_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            daily_change_pct = (daily_change / data['Close'].iloc[-2]) * 100
            st.metric("Daily Change", f"${daily_change:.2f}", f"{daily_change_pct:.2f}%")
        
        with col3:
            st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        
        with col4:
            if st.session_state.last_update:
                st.metric("Last Update", st.session_state.last_update.strftime("%H:%M:%S"))
        
        # Run backtest
        with st.spinner("Running backtest..."):
            backtest_results = run_backtest(data)
        
        if backtest_results:
            # Display backtest results
            st.markdown("## 📊 Backtest Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Starting Value", f"${backtest_results['starting_value']:,.2f}")
            with col2:
                st.metric("Final Value", f"${backtest_results['final_value']:,.2f}")
            with col3:
                return_color = "normal" if backtest_results['return_pct'] >= 0 else "inverse"
                st.metric("Total Return", f"{backtest_results['return_pct']:.2f}%")
        
        # Display chart
        st.markdown("## 📈 Price Chart & Moving Averages")
        chart = create_chart(data)
        st.plotly_chart(chart, use_container_width=True)
        
        # Display recent data
        st.markdown("## 📋 Recent Price Data")
        st.dataframe(data.tail(10))
        
        # Trading signals
        st.markdown("## 🎯 Current Trading Signal")
        sma_50 = data['Close'].rolling(window=short_period).mean().iloc[-1]
        sma_200 = data['Close'].rolling(window=long_period).mean().iloc[-1]
        
        if sma_50 > sma_200:
            st.success("🟢 BULLISH - Short MA above Long MA")
        else:
            st.error("🔴 BEARISH - Short MA below Long MA")
        
        # Live market status
        st.markdown("## 🔴 Live Market Status")
        current_time = datetime.now()
        market_hours = "Market hours: 9:30 AM - 4:00 PM ET (Mon-Fri)"
        st.info(f"Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | {market_hours}")
        
    else:
        st.error("Failed to load market data. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
