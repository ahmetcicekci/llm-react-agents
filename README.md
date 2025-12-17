# llm-react-agents

A ReAct-style financial analysis chatbot powered by LangChain and OpenAI that fetches stock data, calculates financial metrics, and creates visualizations.

## How to run

```bash
uv run react_tools.py
```

## Changes from Ahmet Çiçekci's Version

This version includes significant enhancements made by Necati Sefercioğlu:

### New Features

1. **Enhanced Stock Data Fetching**
   - Renamed `get_stock_prices` → `get_stock_data`
   - Now fetches complete OHLCV data (Open, High, Low, Close, Volume) instead of just closing prices
   - Provides richer dataset for comprehensive financial analysis

2. **Financial Metrics Calculator** (NEW)
   - Added `calculate_financial_metrics` function that computes:
     - Basic statistics (First Open, Last Close, High, Low, Average Volume)
     - Yearly and cumulative returns (%)
     - Annualized volatility (%)
     - Sharpe ratio (using 2% risk-free rate)
     - Maximum drawdown (%)
   - Outputs both CSV and formatted TXT files for easy reading

3. **Interactive Chat Interface**
   - Converted from single-query execution to continuous conversation loop
   - Type 'exit' or 'quit' to end the session
   - Agent introduces itself and explains capabilities at startup
   - Maintains conversation context across multiple queries

4. **Improved Output Formatting**
   - Separates reasoning process from final output
   - Shows "Thinking" section with THOUGHT/ACTION/OBSERVATION/DECISION flow
   - Displays clean "FINAL OUTPUT" section with results
   - Better visualization of the agent's decision-making process

5. **Enhanced System Prompt**
   - Clearer ReAct reasoning structure with uppercase keywords
   - Explicit workflow guidance (must fetch data first)
   - Smart ticker symbol inference (e.g., "Apple" → "AAPL")
   - Domain-specific focus on financial analysis only

### Removed Features

- Removed DuckDuckGo search tool (simplified to focus on financial tools)
- Cleaned up sample output files from repository

### Technical Improvements

- Added `numpy` dependency for financial calculations
- Improved data handling with forward fill for missing values
- Better error handling and validation
- Backward compatibility maintained for plotting function

## Available Tools

1. **get_stock_data** - Fetches historical OHLCV data for any stock ticker
2. **calculate_financial_metrics** - Computes comprehensive financial metrics
3. **plot_stock_prices** - Creates time-series visualizations