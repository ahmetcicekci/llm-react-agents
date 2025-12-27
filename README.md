# FinSight ReAct Agent

A ReAct-style financial analysis chatbot powered by LangGraph and OpenAI that fetches stock data, calculates financial metrics, performs time series analysis, and creates comprehensive visualizations.

## How to Run

```bash
uv run react_tools.py
```

The agent will start an interactive chat session. Type your financial analysis questions, and the agent will use its available tools to help you. Type 'exit' or 'quit' to end the session.

## Features

### Stock Data Analysis
- Fetches complete OHLCV data (Open, High, Low, Close, Volume) for any stock ticker
- Supports historical data retrieval with customizable date ranges
- Automatically handles missing data with forward fill
- Data saved to CSV files for further analysis

### Financial Metrics
Computes comprehensive financial metrics including:
- Basic statistics (First Open, Last Close, High, Low, Average Volume)
- Yearly and cumulative returns (%)
- Annualized volatility (%)
- Sharpe ratio (using 2% risk-free rate)
- Maximum drawdown (%)

Outputs results in both CSV and formatted TXT files for easy analysis.

### Time Series Analysis
- **Decomposition**: Breaks down stock prices into trend, seasonal, and residual components
- **Seasonality Detection**: Identifies repeating patterns in price movements
- Customizable period for seasonal analysis (default: 30 days)

### Price Prediction
- **Random Forest Model**: Predicts future closing prices using machine learning
- Creates lagged features from past 3 days of data
- Iteratively predicts up to 7 days ahead (configurable)
- Visualizes historical prices vs. predicted prices

### Volatility Prediction
- **Linear Regression Model**: Predicts future volatility using rolling windows
- Analyzes both 7-day and 30-day volatility patterns
- Provides actual vs. predicted volatility comparisons
- Outputs confidence intervals and model accuracy metrics

### Visualization
- Time-series plots of stock prices
- Time series decomposition plots (trend, seasonal, residual)
- Price prediction plots with historical context
- Volatility prediction plots with multiple time windows
- All visualizations saved as high-quality PNG files

### Interactive Interface
- Continuous conversation loop for multiple queries
- Agent introduces its capabilities at startup
- Maintains conversation context throughout the session
- Clear separation between reasoning process and final output
- Built with LangGraph for robust state management

### ReAct Reasoning
- Transparent thinking process with structured reasoning
- Smart ticker symbol inference (e.g., "Apple" → "AAPL")
- Structured workflow that ensures data is fetched before analysis
- Domain-specific focus on financial analysis
- Powered by OpenAI GPT models (configurable to use Google Gemini)

## Available Tools

1. **get_stock_data** - Fetches historical OHLCV data for any stock ticker with customizable date ranges
2. **calculate_financial_metrics** - Computes comprehensive financial metrics and saves results to CSV
3. **display_financial_metrics** - Displays formatted financial metrics summary in the terminal
4. **plot_stock_prices** - Creates time-series visualizations of stock closing prices
5. **decompose_time_series** - Performs seasonal decomposition and creates visualization
6. **predict_future_prices** - Uses Random Forest to predict future closing prices
7. **predict_volatility** - Uses Linear Regression to predict future volatility

## Requirements

- Python 3.11+
- Dependencies managed via `uv`:
  - LangChain & LangGraph (agent framework)
  - OpenAI & Google Gemini (LLM providers)
  - yfinance (stock data)
  - pandas & numpy (data processing)
  - matplotlib (visualizations)
  - statsmodels (time series decomposition)
  - scikit-learn (machine learning models)
  - python-dotenv (environment variables)

## Example Outputs

The agent creates organized folders for each analysis session (e.g., `GOOGLE 2015-2023/`, `TESLA 2020-2024/`) containing:
- Raw OHLCV data (CSV)
- Financial metrics (CSV and TXT)
- Price plots (PNG)
- Decomposition plots (PNG)
- Price predictions (CSV and PNG)
- Volatility analysis (CSV and PNG)

See the `Edge Cases/` folder for examples of how the agent handles out-of-domain requests.
