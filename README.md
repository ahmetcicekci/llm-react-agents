# llm-react-agents

## How to run

```bash
uv run react_tools.py
```

## Sample output 

```text
================================ Human Message =================================

Show me google stock prices between 2024 and 2025.
================================== Ai Message ==================================

Thought: I will fetch the historical daily closing prices for Alphabet (GOOGL) for the years 2024 through 2025, then plot them to visualize the trend.
Action: call the get_stock_prices tool with ticker "GOOGL", start_year 2024, end_year 2025.
Tool Calls:
  get_stock_prices (call_dks1nK9c8mS1DVoiXnsaNzl8)
 Call ID: call_dks1nK9c8mS1DVoiXnsaNzl8
  Args:
    ticker: GOOGL
    start_year: 2024
    end_year: 2025
================================= Tool Message =================================
Name: get_stock_prices

{"message": "Saved prices to GOOGL_2024_to_2025.csv", "csv_file": "GOOGL_2024_to_2025.csv"}
================================== Ai Message ==================================

Observation: Retrieved CSV file path GOOGL_2024_to_2025.csv containing Google stock prices for 2024-2025.
Thought: I will plot the stock prices to visualize the data.
Action: call the plot_stock_prices tool with csv_file set to "GOOGL_2024_to_2025.csv".
Tool Calls:
  plot_stock_prices (call_lQtcOAzb6f2ZV1ZubQQjhaP0)
 Call ID: call_lQtcOAzb6f2ZV1ZubQQjhaP0
  Args:
    csv_file: GOOGL_2024_to_2025.csv
================================= Tool Message =================================
Name: plot_stock_prices

Plot saved to GOOGL_2024_to_2025_plot.png
================================== Ai Message ==================================

Here are the Google (GOOGL) stock prices for 2024 and 2025:

- Data range: 2024-01-01 to 2025-12-31
- Data source: Historical daily closing prices
- File with data: GOOGL_2024_to_2025.csv
- Plot image: GOOGL_2024_to_2025_plot.png

Would you like me to summarize the data (e.g., annual returns, average price, volatility) or export the data to another format (CSV, Excel)?
```