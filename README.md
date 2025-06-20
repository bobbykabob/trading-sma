# Real-Time Trading Strategy Dashboard

A web-based dashboard for running and visualizing multiple trading strategies in real-time using yfinance data.

## 🌐 Live Demo

**🎯 Try it now**: [https://harris-song-trading-sma.streamlit.app/](https://harris-song-trading-sma.streamlit.app/)

## Features

- **Multi-Market Analysis**: ES=F, SPY, QQQ, BTC-USD, GLD, TLT - choose up to 4 markets
- **Dual Trading Strategies**: SMA Crossover + Breakout Strategy with volume confirmation
- **Real-time Intraday Data**: 1m, 2m, 5m, 15m, 30m, 1h, 1d intervals
- **Interactive UI**: Beautiful web dashboard with live charts and metrics
- **Strategy Backtesting**: Performance metrics with customizable parameters
- **Auto-refresh**: Optional 30-second auto-refresh for continuous monitoring
- **Visual Charts**: Interactive candlestick charts with moving averages, volume, and breakout signals

## Quick Start

### Option 1: Try the Live Demo
Visit: [https://harris-song-trading-sma.streamlit.app/](https://harris-song-trading-sma.streamlit.app/)

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/bobbykabob/trading-sma.git
cd trading-sma

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

### Option 3: Using the Launcher
```bash
python launcher.py
```

## How It Works

1. **Data Fetching**: Uses yfinance to get real-time market data for multiple markets
2. **Dual Strategies**: 
   - **SMA Crossover**: Moving Average crossover signals (customizable periods)
   - **Breakout Strategy**: Support/resistance breakouts with volume confirmation
3. **Backtesting**: Runs strategies on historical data using Backtrader
4. **Visualization**: Displays results in an interactive multi-panel dashboard

## Dashboard Features

### 📊 **Multi-Market Overview**
- **6 Markets**: ES=F (S&P Futures), SPY (S&P ETF), QQQ (Nasdaq), BTC-USD (Bitcoin), GLD (Gold), TLT (Bonds)
- **Market Selection**: Choose up to 4 markets for simultaneous analysis
- **Real-time Metrics**: Current price, daily change, volume, and last update time

### 📈 **Trading Strategies**

#### **1. SMA Crossover Strategy**
- **Customizable Periods**: Short (5-300) and Long (5-300) moving averages
- **Real-time Signals**: 
  - 🟢 **BULLISH** - Short MA above Long MA
  - 🔴 **BEARISH** - Short MA below Long MA
- **Performance Tracking**: Starting value, final value, total return percentage

#### **2. Breakout Strategy**
- **Support/Resistance**: Dynamic levels based on lookback period (5-100 days)
- **Volume Confirmation**: Customizable volume threshold (1.0x - 3.0x average)
- **Risk Management**: Stop-loss (1-20%) and take-profit (1-50%) settings
- **Real-time Signals**:
  - 🚀 **BREAKOUT** - Price above resistance with volume confirmation
  - 💥 **BREAKDOWN** - Price below support with volume confirmation  
  - ⏳ **CONSOLIDATION** - Price trading between support/resistance levels

### 📊 **Interactive Charts**
- **3-Panel Layout**: Price/Moving Averages, Volume Analysis, Breakout Signals
- **Intraday Data**: 1m, 2m, 5m, 15m, 30m, 1h, 1d intervals
- **Support/Resistance Lines**: Dynamic resistance and support levels
- **Volume Analysis**: Color-coded volume bars with moving average overlay
- **Auto-refresh**: 30-second updates with countdown timer

## Strategy Details

### **SMA Crossover Strategy**
- **Default Periods**: 20/50 (intraday) or 50/200 (daily)
- **Buy Signal**: When short MA crosses above long MA
- **Sell Signal**: When short MA crosses below long MA
- **Starting Capital**: $10,000
- **Commission**: 0.1% per trade

### **Breakout Strategy**
- **Support/Resistance**: Calculated from lookback period (default 20)
- **Entry Conditions**: Price breakout + volume > threshold
- **Risk Management**: Automatic stop-loss and take-profit levels
- **Volume Confirmation**: Prevents false breakouts

## Files

- `app.py` - Main Streamlit dashboard application
- `main.py` - Original backtrader strategy (standalone)
- `launcher.py` - Easy setup and launch script
- `run_app.sh` - Shell script to install and run
- `requirements.txt` - Python dependencies

## Usage Tips

1. **Live Demo**: Visit [https://harris-song-trading-sma.streamlit.app/](https://harris-song-trading-sma.streamlit.app/) for instant access
2. **First Run**: The app will fetch intraday data (may take a moment for multiple markets)
3. **Auto-refresh**: Enable for continuous monitoring during market hours
4. **Strategy Tuning**: Adjust all parameters in the sidebar to test different setups
5. **Market Hours**: Most relevant during market trading hours
6. **Intraday Trading**: Use 5m-15m intervals for day trading, 1h-1d for swing trading

## Requirements

- Python 3.7+
- Internet connection for data fetching
- All dependencies listed in requirements.txt

## Stopping the App

Press `Ctrl+C` in the terminal to stop the dashboard.

## 🚀 Deployment

### Streamlit Cloud (Recommended)

**✅ Already Deployed**: [https://harris-song-trading-sma.streamlit.app/](https://harris-song-trading-sma.streamlit.app/)

To deploy your own version:

1. **Push to GitHub**: Make sure your code is pushed to GitHub
2. **Visit Streamlit Cloud**: Go to [share.streamlit.io](https://share.streamlit.io)
3. **Connect GitHub**: Sign in with your GitHub account
4. **Deploy**: Select your repository `bobbykabob/trading-sma` and set the main file as `app.py`
5. **Auto-deploy**: Your app will be live and auto-update on every push!

### AWS EC2 Deployment

1. **Launch EC2 Instance**: Use Ubuntu 20.04 LTS
2. **Install Dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip
   git clone https://github.com/bobbykabob/trading-sma.git
   cd trading-sma
   pip3 install -r requirements.txt
   ```
3. **Run with PM2**:
   ```bash
   sudo npm install -g pm2
   pm2 start "streamlit run app.py --server.port 8501" --name trading-dashboard
   ```
4. **Setup Nginx** (optional): Configure reverse proxy for custom domain

### Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📊 What Makes This Special

- **Multi-Strategy Approach**: Combines trend-following (SMA) with momentum (Breakout) strategies
- **Real-time Intraday Data**: True minute-by-minute market data for active trading
- **Multi-Market Analysis**: Compare stocks, futures, crypto, and bonds simultaneously
- **Professional Risk Management**: Built-in stop-loss and take-profit levels
- **Volume Confirmation**: Reduces false signals with volume analysis
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 🎯 Perfect For

- **Day Traders**: 5-minute intraday breakout signals
- **Swing Traders**: Daily SMA crossover strategies  
- **Multi-Asset Traders**: Compare different market sectors
- **Algorithm Developers**: Test and refine trading strategies
- **Market Analysts**: Real-time market monitoring and analysis

---

**🌐 Live Demo**: [https://harris-song-trading-sma.streamlit.app/](https://harris-song-trading-sma.streamlit.app/)

**📧 Questions?** Open an issue on GitHub or try the live demo!
