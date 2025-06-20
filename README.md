# Real-Time Trading Strategy Dashboard

A web-based dashboard for running and visualizing trading strategies in real-time using yfinance data.

## Features

- **Real-time Data**: Automatically fetches the latest ES=F (E-mini S&P 500 futures) data
- **Interactive UI**: Beautiful web dashboard with live charts and metrics
- **Strategy Backtesting**: Moving Average Crossover strategy with performance metrics
- **Auto-refresh**: Optional 30-second auto-refresh for continuous monitoring
- **Visual Charts**: Interactive candlestick charts with moving averages and volume

## Quick Start

### Option 1: Using the Launcher (Recommended)
```bash
python launcher.py
```

### Option 2: Using the Shell Script
```bash
./run_app.sh
```

### Option 3: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## How It Works

1. **Data Fetching**: Uses yfinance to get real-time market data for ES=F futures
2. **Strategy Execution**: Implements a Moving Average Crossover strategy (50-day vs 200-day SMA)
3. **Backtesting**: Runs the strategy on historical data using Backtrader
4. **Visualization**: Displays results in an interactive web dashboard

## Dashboard Features

- **Live Metrics**: Current price, daily change, volume, and last update time
- **Backtest Results**: Starting value, final value, and total return percentage
- **Interactive Charts**: Candlestick price chart with moving averages and volume
- **Trading Signals**: Current bullish/bearish signal based on MA crossover
- **Auto-refresh**: Toggle 30-second auto-refresh for continuous monitoring
- **Strategy Parameters**: Adjustable MA periods via sidebar controls

## Strategy Details

The system uses a simple Moving Average Crossover strategy:
- **Buy Signal**: When 50-day SMA crosses above 200-day SMA
- **Sell Signal**: When 50-day SMA crosses below 200-day SMA
- **Starting Capital**: $10,000
- **Commission**: 0.1% per trade

## Files

- `app.py` - Main Streamlit dashboard application
- `main.py` - Original backtrader strategy (standalone)
- `launcher.py` - Easy setup and launch script
- `run_app.sh` - Shell script to install and run
- `requirements.txt` - Python dependencies

## Usage Tips

1. **First Run**: The app will fetch several years of historical data (may take a moment)
2. **Auto-refresh**: Enable for continuous monitoring during market hours
3. **Strategy Tuning**: Adjust MA periods in the sidebar to test different parameters
4. **Market Hours**: Most relevant during futures trading hours (almost 24/5)

## Requirements

- Python 3.7+
- Internet connection for data fetching
- All dependencies listed in requirements.txt

## Stopping the App

Press `Ctrl+C` in the terminal to stop the dashboard.

## 🚀 Deployment

### Streamlit Cloud (Recommended)

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

## 📊 Live Demo

Once deployed on Streamlit Cloud, your app will be available at:
`https://bobbykabob-trading-sma-app-xyz123.streamlit.app`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request
