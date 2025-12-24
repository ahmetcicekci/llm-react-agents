from dotenv import load_dotenv

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import os

load_dotenv()


def get_stock_data(ticker: str, start_year: int = None, end_year: int = None) -> str:
    """
    Fetches historical daily OHLCV (Open, High, Low, Close, Volume) data
    for a given stock ticker between two years. Saves the results as a CSV file.

    Args:
        ticker (str): Stock ticker symbol.
        start_year (int): Starting year (e.g., 2022).
        end_year (int): Ending year (e.g., 2025).

    Returns:
        dict: Status message and path of the saved CSV file.
    """

    # Default values if user did not provide years
    if start_year is None:
        start_year = 2022
    if end_year is None:
        end_year = 2025

    # Construct date range
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-01-01"

    # Fetch price history
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval="1d")

    if df.empty:
        return f"No price data found for {ticker} between {start_year} and {end_year}."

    # Keep OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Save file
    filename = f"{ticker}_{start_year}_to_{end_year}.csv"
    df.to_csv(filename)

    return {"message": f"Saved stock data to {filename}", "csv_file": filename}


def calculate_financial_metrics(csv_file: str) -> dict:
    """
    Calculates comprehensive financial metrics from OHLCV data.

    Computes per-year and cumulative metrics including:
    - Basic statistics (First Open, Last Close, High, Low, Avg Volume)
    - Yearly and cumulative returns (%)
    - Annualized volatility (%)
    - Sharpe ratio (using 2% risk-free rate)
    - Maximum drawdown (%)

    Args:
        csv_file (str): Path to CSV file with OHLCV data (output from get_stock_data)

    Returns:
        dict: Status message and path to metrics CSV file
    """

    try:
        # Load data
        df = pd.read_csv(csv_file)

        # Convert Date column to datetime explicitly and remove timezone info
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)

        # Validate required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {"message": f"Error: Missing required columns: {missing_cols}", "csv_file": None}

        # Handle missing data with forward fill (updated method)
        df = df.ffill()

        # Extract year from dates
        df["Year"] = df["Date"].dt.year

        # Helper function to calculate metrics for a dataframe
        def calc_metrics(data, period_name):
            if len(data) == 0:
                return None

            first_open = data.iloc[0]["Open"]
            last_close = data.iloc[-1]["Close"]
            high = data["High"].max()
            low = data["Low"].min()
            avg_volume = data["Volume"].mean()

            # Yearly return (%)
            yearly_return = ((last_close - first_open) / first_open) * 100

            # Annualized volatility (%)
            daily_returns = data["Close"].pct_change().dropna()
            if len(daily_returns) > 1:
                volatility = daily_returns.std() * np.sqrt(252) * 100
            else:
                volatility = 0.0

            # Sharpe ratio (using 2% risk-free rate)
            risk_free_rate = 2.0
            if volatility > 0:
                sharpe_ratio = (yearly_return - risk_free_rate) / volatility
            else:
                sharpe_ratio = np.nan

            # Maximum drawdown (%)
            cumulative = (1 + data["Close"].pct_change()).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100

            return {
                "Period": period_name,
                "First_Open": round(first_open, 2),
                "Last_Close": round(last_close, 2),
                "High": round(high, 2),
                "Low": round(low, 2),
                "Avg_Volume": round(avg_volume, 2),
                "Yearly_Return_%": round(yearly_return, 2),
                "Annualized_Volatility_%": round(volatility, 2),
                "Sharpe_Ratio": round(sharpe_ratio, 3) if not np.isnan(sharpe_ratio) else "N/A",
                "Max_Drawdown_%": round(max_drawdown, 2)
            }

        # Calculate metrics per year
        metrics_list = []
        for year, year_df in df.groupby("Year"):
            metrics = calc_metrics(year_df, str(year))
            if metrics:
                metrics_list.append(metrics)

        # Calculate cumulative metrics
        cumulative_metrics = calc_metrics(df, "Cumulative")
        if cumulative_metrics:
            metrics_list.append(cumulative_metrics)

        # Create output DataFrame
        metrics_df = pd.DataFrame(metrics_list)

        # Generate output filenames from input filename
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        csv_output = f"{base_name}_metrics.csv"
        txt_output = f"{base_name}_metrics.txt"

        # Save to CSV
        metrics_df.to_csv(csv_output, index=False)

        # Save formatted table to TXT file
        with open(txt_output, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("FINANCIAL METRICS SUMMARY\n")
            f.write("=" * 100 + "\n")
            f.write(metrics_df.to_string(index=False) + "\n")
            f.write("=" * 100 + "\n")

        return {
            "message": f"Calculated metrics saved to {csv_output} and {txt_output}",
            "csv_file": csv_output
        }

    except FileNotFoundError:
        return {"message": f"Error: Could not find file {csv_file}", "csv_file": None}
    except Exception as e:
        return {"message": f"Error calculating metrics: {str(e)}", "csv_file": None}


def display_financial_metrics(csv_file: str) -> str:
    """
    Reads a metrics CSV file and RETURNS it as a formatted table string.
    No printing, no side effects.
    """
    df = pd.read_csv(csv_file)

    if df.empty:
        return "Metrics file is empty."

    output = "\n" + "=" * 100 + "\n"
    output += "FINANCIAL METRICS SUMMARY\n"
    output += "=" * 100 + "\n"
    output += df.to_string(index=False)
    output += "\n" + "=" * 100 + "\n"

    return output


def plot_stock_prices(csv_file: str) -> str:
    """
    Plots the time-series contained in a CSV file.
    If OHLCV data exists, plots Close prices.
    Maintains backward compatibility with old format.
    Saves plot as {ticker}_plot.png.
    """
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    # Determine which column to plot
    if "Close" in df.columns:
        y_col = "Close"
        y_label = "Close Price"
    else:
        # Fallback for old format - use first numeric column
        y_col = df.columns[1]
        y_label = "Price"

    # Derive plot filename from csv filename
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    plot_file = f"{base_name}_plot.png"

    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], df[y_col])
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.title(f"{y_label}s from {base_name}")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_file)
    plt.close()
    return f"Plot saved to {plot_file}"


def decompose_time_series(csv_file: str, period: int = None) -> str:
    """
    Performs time series decomposition on stock price data and creates a visualization.

    Decomposes the time series into:
    - Trend component (long-term progression)
    - Seasonal component (repeating patterns)
    - Residual component (irregular fluctuations)

    Args:
        csv_file (str): Path to CSV file with OHLCV data (output from get_stock_data)
        period (int): Seasonal period for decomposition. If None, defaults to 30 (monthly pattern)

    Returns:
        str: Status message with path to saved decomposition plot
    """
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    # Determine which column to decompose
    if "Close" in df.columns:
        y_col = "Close"
    else:
        # Fallback for old format - use first numeric column
        y_col = df.columns[1]

    # Set index to Date for decomposition
    df = df.set_index("Date")

    # Default period to 30 days (approximately monthly seasonality) if not specified
    if period is None:
        period = 30

    # Ensure we have enough data points for decomposition
    if len(df) < 2 * period:
        return f"Error: Not enough data points for decomposition. Need at least {2 * period} data points, but only have {len(df)}."

    # Perform seasonal decomposition
    decomposition = seasonal_decompose(df[y_col], model='additive', period=period)

    # Derive plot filename from csv filename
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    plot_file = f"{base_name}_decomposition.png"

    # Create decomposition plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Original
    decomposition.observed.plot(ax=axes[0])
    axes[0].set_ylabel('Observed')
    axes[0].set_title(f'Time Series Decomposition - {base_name}')
    axes[0].grid(True)

    # Trend
    decomposition.trend.plot(ax=axes[1])
    axes[1].set_ylabel('Trend')
    axes[1].grid(True)

    # Seasonal
    decomposition.seasonal.plot(ax=axes[2])
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True)

    # Residual
    decomposition.resid.plot(ax=axes[3])
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    return f"Time series decomposition plot saved to {plot_file}"


def predict_future_prices(csv_file: str, days_ahead: int = 7) -> dict:
    """
    Predicts future stock closing prices using a Random Forest model.
    Creates lagged features from the last 3 days and iteratively predicts future prices.
    Saves predictions to CSV and creates a visualization plot.

    Args:
        csv_file (str): Path to CSV file with OHLCV data (output from get_stock_data)
        days_ahead (int): Number of days to predict into the future (default: 7)

    Returns:
        dict: Status message, prediction CSV file path, and plot file path
    """
    try:
        # Load data
        df = pd.read_csv(csv_file, parse_dates=["Date"])

        if "Close" not in df.columns:
            return {
                "message": "Error: CSV does not contain Close column",
                "csv_file": None,
                "plot_file": None
            }

        # Sort by date to ensure chronological order
        df = df.sort_values("Date")

        # Create lagged features (use past 3 days to predict current day)
        df["Close_lag1"] = df["Close"].shift(1)
        df["Close_lag2"] = df["Close"].shift(2)
        df["Close_lag3"] = df["Close"].shift(3)
        df = df.dropna()

        if len(df) < 10:
            return {
                "message": "Error: Not enough data points to train prediction model (need at least 10)",
                "csv_file": None,
                "plot_file": None
            }

        # Prepare features and target
        X = df[["Close_lag1", "Close_lag2", "Close_lag3"]]
        y = df["Close"]

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)

        # Get last 3 closing prices for prediction
        last_row = df.tail(1)
        close_l1 = last_row["Close_lag1"].values[0]
        close_l2 = last_row["Close_lag2"].values[0]
        close_l3 = last_row["Close_lag3"].values[0]

        # Iteratively predict future prices
        predictions = []
        for _ in range(days_ahead):
            # Create DataFrame with proper column names to avoid sklearn warning
            X_pred = pd.DataFrame([[close_l1, close_l2, close_l3]],
                                   columns=["Close_lag1", "Close_lag2", "Close_lag3"])
            pred = float(model.predict(X_pred)[0])
            predictions.append(pred)

            # Shift the window forward
            close_l3 = close_l2
            close_l2 = close_l1
            close_l1 = pred

        # Save predictions to CSV
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        pred_file = f"{base_name}_price_predictions.csv"

        pred_df = pd.DataFrame({
            "Day": list(range(1, days_ahead + 1)),
            "Predicted_Close": predictions
        })
        pred_df.to_csv(pred_file, index=False)

        # Create visualization
        plot_file = f"{base_name}_price_predictions_plot.png"

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical prices (last 30 days)
        historical = df.tail(30)
        ax.plot(historical["Date"], historical["Close"],
                label="Historical Prices", color="blue", linewidth=2)

        # Plot predictions
        last_date = df["Date"].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                      periods=days_ahead, freq='D')
        ax.plot(future_dates, predictions,
                label="Predicted Prices", color="red", linewidth=2, linestyle="--", marker='o')

        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.set_title(f"Price Prediction - {base_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(plot_file)
        plt.close()

        return {
            "message": f"Price predictions saved to {pred_file} and plot saved to {plot_file}",
            "csv_file": pred_file,
            "plot_file": plot_file
        }

    except Exception as e:
        return {
            "message": f"Error in price prediction: {str(e)}",
            "csv_file": None,
            "plot_file": None
        }


def predict_volatility(csv_file: str) -> dict:
    """
    Predicts future volatility using rolling volatility windows and Linear Regression.
    Computes last 7-day volatility and predicts next-day volatility.
    Creates plots showing actual vs predicted volatility for both 7-day and 30-day windows.
    Saves summary to CSV and creates a visualization plot.

    Args:
        csv_file (str): Path to CSV file with OHLCV data (output from get_stock_data)

    Returns:
        dict: Status message, volatility CSV file path, and plot file path
    """
    try:
        # Load data
        df = pd.read_csv(csv_file, parse_dates=["Date"])

        if "Close" not in df.columns:
            return {
                "message": "Error: CSV missing Close column",
                "csv_file": None,
                "plot_file": None
            }

        # Sort by date
        df = df.sort_values("Date")

        # Calculate daily returns
        df["Return"] = df["Close"].pct_change()

        # Calculate rolling volatilities
        df["Volatility_7d"] = df["Return"].rolling(7).std()
        df["Volatility_30d"] = df["Return"].rolling(30).std()

        # Get last 7-day volatility
        last_7d_vol = df["Volatility_7d"].iloc[-1]
        last_7d_vol = float(last_7d_vol) if not np.isnan(last_7d_vol) else None

        # Prepare data for prediction model
        df_clean = df.dropna()

        if len(df_clean) < 40:
            return {
                "message": "Error: Not enough data to train volatility model (need at least 40 days)",
                "csv_file": None,
                "plot_file": None
            }

        # Split data: use 80% for training, 20% for testing
        split_idx = int(len(df_clean) * 0.8)
        df_train = df_clean.iloc[:split_idx].copy()
        df_test = df_clean.iloc[split_idx:].copy()

        # Create features and targets for 7-day volatility prediction
        X_train_7d = df_train[["Volatility_7d", "Volatility_30d"]]
        y_train_7d = df_train["Volatility_7d"].shift(-1).dropna()
        X_train_7d = X_train_7d.iloc[:-1]

        # Create features and targets for 30-day volatility prediction
        X_train_30d = df_train[["Volatility_7d", "Volatility_30d"]]
        y_train_30d = df_train["Volatility_30d"].shift(-1).dropna()
        X_train_30d = X_train_30d.iloc[:-1]

        # Train models
        model_7d = LinearRegression()
        model_7d.fit(X_train_7d, y_train_7d)

        model_30d = LinearRegression()
        model_30d.fit(X_train_30d, y_train_30d)

        # Make predictions on test set
        X_test = df_test[["Volatility_7d", "Volatility_30d"]]

        # Use DataFrame with column names to avoid sklearn warning
        pred_7d = model_7d.predict(X_test)
        pred_30d = model_30d.predict(X_test)

        # Add predictions to test dataframe
        df_test["Predicted_7d"] = pred_7d
        df_test["Predicted_30d"] = pred_30d

        # Predict next day's volatility using full dataset
        X_last = df_clean[["Volatility_7d", "Volatility_30d"]].iloc[[-1]]
        next_vol_7d = float(model_7d.predict(X_last)[0])
        next_vol_30d = float(model_30d.predict(X_last)[0])

        # Save volatility summary to CSV
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        vol_file = f"{base_name}_volatility_summary.csv"

        vol_df = pd.DataFrame({
            "Last_7d_Volatility": [last_7d_vol],
            "Predicted_Next_7d_Volatility": [next_vol_7d],
            "Predicted_Next_30d_Volatility": [next_vol_30d]
        })
        vol_df.to_csv(vol_file, index=False)

        # Create visualization
        plot_file = f"{base_name}_volatility_plot.png"

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: Actual vs Predicted 7-Day Volatility
        ax1.plot(df_test["Date"], df_test["Volatility_7d"],
                 label="Actual 7-Day Volatility", color="blue", linewidth=2)
        ax1.plot(df_test["Date"], df_test["Predicted_7d"],
                 label="Predicted 7-Day Volatility", color="red", linewidth=2, linestyle="--", alpha=0.7)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Volatility")
        ax1.set_title(f"7-Day Volatility: Actual vs Predicted - {base_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Actual vs Predicted 30-Day Volatility
        ax2.plot(df_test["Date"], df_test["Volatility_30d"],
                 label="Actual 30-Day Volatility", color="orange", linewidth=2)
        ax2.plot(df_test["Date"], df_test["Predicted_30d"],
                 label="Predicted 30-Day Volatility", color="purple", linewidth=2, linestyle="--", alpha=0.7)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volatility")
        ax2.set_title(f"30-Day Volatility: Actual vs Predicted - {base_name}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Close prices for context
        ax3.plot(df["Date"], df["Close"], color="green", linewidth=1.5)
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Close Price")
        ax3.set_title(f"Close Prices - {base_name}")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()

        return {
            "message": f"Volatility analysis saved to {vol_file} and plot saved to {plot_file}",
            "csv_file": vol_file,
            "plot_file": plot_file
        }

    except Exception as e:
        return {
            "message": f"Error in volatility prediction: {str(e)}",
            "csv_file": None,
            "plot_file": None
        }


tools = [get_stock_data, calculate_financial_metrics, display_financial_metrics, plot_stock_prices, decompose_time_series, predict_future_prices, predict_volatility]
llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0)  # swap with ChatGoogleGenerativeAI(...) to use Gemini
llm_with_tools = llm.bind_tools(tools)


# Node
def reasoner(state: MessagesState):
    sys_msg = SystemMessage(
        content=(
            "You are a financial analysis assistant that ONLY handles questions about stock market data, "
            "financial metrics, and investment analysis. Reject any requests outside this domain.\n\n"

            "When first greeted or at the start of a conversation, introduce yourself and briefly describe "
            "your capabilities: you can fetch historical stock data, calculate financial metrics "
            "(returns, volatility, Sharpe ratio, maximum drawdown), create price visualizations, "
            "perform time series decomposition to identify trends and patterns, predict future prices, "
            "and forecast volatility.\n\n"

            "MANDATORY WORKFLOW:\n"
            "For new stock analysis requests, the first step must be fetching stock data. "
            "If the user mentions a company name (e.g., 'Apple', 'Google'), infer the ticker symbol (e.g., 'AAPL', 'GOOGL'). "
            "If no year range is specified, use default values (2022-2025).\n\n"

            "Exception: If the user references an existing CSV file by name, you may work directly with that file.\n\n"

            "After fetching stock data, you can:\n"
            "- Calculate comprehensive financial metrics (returns, volatility, risk metrics)\n"
            "- Display calculated metrics in a formatted table\n"
            "- Create visualizations of historical price movements\n"
            "- Decompose time series into trend, seasonal, and residual components\n"
            "- Generate future price predictions with visual forecasts\n"
            "- Forecast volatility and risk with trend analysis\n\n"

            "Use ReAct style reasoning in a loop with the following format:\n"
            "THOUGHT: Describe what you need to do\n"
            "ACTION: Call the appropriate tool\n"
            "OBSERVATION: Interpret the tool's output\n"
            "DECISION: Either continue to the next action OR provide the final answer to the user\n\n"
            "Always output each step on a new line with the keywords in UPPERCASE.\n"
            "- THOUGHT, ACTION, OBSERVATION, and DECISION steps are INTERNAL reasoning.\n"
            "- NEVER include THOUGHT, ACTION, OBSERVATION, or DECISION text in the FINAL OUTPUT.\n"
            "FINAL OUTPUT RULES:\n"
            "- Keep the final answer under 4 lines.\n"
            "- Do NOT repeat the step-by-step workflow in the final answer.\n"
            "- The FINAL OUTPUT must NOT mention tools, functions, reasoning steps, or decisions.\n"
            "- Do NOT repeat any data, tables, or metrics that are already displayed above.\n"
            "- Instead, briefly summarize what you did and suggest helpful next steps.\n"
        )
    )

    ai_msg = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [ai_msg]}


builder = StateGraph(MessagesState)
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "reasoner")
builder.add_conditional_edges("reasoner", tools_condition)
builder.add_edge("tools", "reasoner")
react_graph = builder.compile()


# Interactive Chat Loop
if __name__ == "__main__":
    print()
    print("=" * 80)
    print("Financial Analysis Chatbot")
    print("=" * 80)
    print("Type 'exit' or 'quit' to end the conversation.\n")

    messages = []

    # Send initial greeting to trigger agent introduction
    initial_response = react_graph.invoke({"messages": [HumanMessage(content="Hello")]})
    messages = initial_response["messages"]

    # Display the agent's introduction
    print(f"Assistant: {messages[-1].content}\n")
    print("-" * 80 + "\n")

    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Check for exit commands
        if user_input.lower() in ["exit", "quit"]:
            print("\nGoodbye!\n")
            break

        # Skip empty inputs
        if not user_input:
            continue

        # Add user message to history
        messages.append(HumanMessage(content=user_input))

        # Invoke the agent
        response = react_graph.invoke({"messages": messages})

        # Get new messages since last turn
        new_messages = response["messages"][len(messages):]

        # Update message history with full conversation
        messages = response["messages"]

        # Separate reasoning from final output
        reasoning_msgs = []
        final_output = None
        metrics_table = None

        for i, msg in enumerate(new_messages):
            # If AI message with tool calls, it's reasoning
            if msg.type == "ai" and hasattr(msg, 'tool_calls') and msg.tool_calls:
                reasoning_msgs.append(msg)
            # If tool message, it's part of reasoning
            elif msg.type == "tool":
                if isinstance(msg.content, str) and "FINANCIAL METRICS SUMMARY" in msg.content:
                    metrics_table = msg.content
                else:
                    reasoning_msgs.append(msg)
            # Last AI message without tool calls is final output
            elif msg.type == "ai" and i == len(new_messages) - 1:
                final_output = msg
            # Other AI messages in the middle might be intermediate thoughts
            elif msg.type == "ai":
                reasoning_msgs.append(msg)

        # Display reasoning process (with ReAct format: THOUGHT/ACTION/OBSERVATION/DECISION)
        if reasoning_msgs:
            print("\n" + "=" * 80)
            print("Agent Reasoning:")
            print("=" * 80 + "\n")
            for msg in reasoning_msgs:
                # Display AI reasoning content (includes THOUGHT and DECISION)
                if hasattr(msg, 'content') and msg.content:
                    if msg.type == "ai":
                        print(msg.content)
                        print()
                    elif msg.type == "tool":
                        # Tool results are observations
                        print(f"OBSERVATION: {msg.content}")
                        print()

                # Display tool calls as actions
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        print(f"ACTION: {tool_call['name']}({tool_call['args']})")
                        print()

        # Display final output separator
        if final_output and final_output.content:
            print("FINAL OUTPUT:")
            if metrics_table:
                print(metrics_table)
            print("=" * 80, "\n")
            print(f"\n{final_output.content}\n")
            print("=" * 80, "\n")

        print("\n")
