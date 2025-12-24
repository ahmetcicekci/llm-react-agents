# LLM ReAct Agents

A ReAct-style financial analysis chatbot powered by LangChain and OpenAI that fetches stock data, calculates financial metrics, and creates visualizations.

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

### Financial Metrics
Computes comprehensive financial metrics including:
- Basic statistics (First Open, Last Close, High, Low, Average Volume)
- Yearly and cumulative returns (%)
- Annualized volatility (%)
- Sharpe ratio (using 2% risk-free rate)
- Maximum drawdown (%)

Outputs results in both CSV and formatted TXT files for easy analysis.

### Visualization
- Creates time-series plots of stock prices
- Generates PNG files with clean, professional formatting
- Supports visualization of price trends over time

### Interactive Interface
- Continuous conversation loop for multiple queries
- Agent introduces its capabilities at startup
- Maintains conversation context throughout the session
- Clear separation between reasoning process and final output

### ReAct Reasoning
- Transparent thinking process with THOUGHT/ACTION/OBSERVATION/DECISION flow
- Smart ticker symbol inference (e.g., "Apple" → "AAPL")
- Structured workflow that ensures data is fetched before analysis
- Domain-specific focus on financial analysis

## Available Tools

1. **get_stock_data** - Fetches historical OHLCV data for any stock ticker
2. **calculate_financial_metrics** - Computes comprehensive financial metrics and saves results
3. **plot_stock_prices** - Creates time-series visualizations of stock prices

## Requirements

- Python 3.x
- Dependencies: LangChain, OpenAI, yfinance, pandas, numpy, matplotlib