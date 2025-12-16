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


tools = [get_stock_data, calculate_financial_metrics, plot_stock_prices]
llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)  # swap with ChatGoogleGenerativeAI(...) to use Gemini
llm_with_tools = llm.bind_tools(tools)


# Node
def reasoner(state: MessagesState):
    sys_msg = SystemMessage(
        content=(
            "You are a financial analysis assistant that ONLY handles questions about stock market data, "
            "financial metrics, and investment analysis. Reject any requests outside this domain.\n\n"

            "When first greeted or at the start of a conversation, introduce yourself and briefly describe "
            "your capabilities: you can fetch historical stock data, calculate financial metrics "
            "(returns, volatility, Sharpe ratio, maximum drawdown), and create price visualizations.\n\n"

            "MANDATORY WORKFLOW:\n"
            "For new stock analysis requests, the first step must be calling get_stock_data to fetch stock data. "
            "If the user mentions a company name (e.g., 'Apple', 'Google'), infer the ticker symbol (e.g., 'AAPL', 'GOOGL'). "
            "If no year range is specified, the tool will use its default values (2022-2025).\n\n"

            "Exception: If the user references an existing CSV file by name, you may work directly with that file.\n\n"

            "After fetching stock data, you have access to tools for calculating financial metrics and "
            "creating visualizations. Decide which tools to use based on the user's specific request.\n\n"

            "Use ReAct style reasoning in a loop with the following format:\n"
            "THOUGHT: Describe what you need to do\n"
            "ACTION: Call the appropriate tool\n"
            "OBSERVATION: Interpret the tool's output\n"
            "DECISION: Either continue to the next action OR provide the final answer to the user\n\n"
            "Always output each step on a new line with the keywords in UPPERCASE.\n"
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

        for i, msg in enumerate(new_messages):
            # If AI message with tool calls, it's reasoning
            if msg.type == "ai" and hasattr(msg, 'tool_calls') and msg.tool_calls:
                reasoning_msgs.append(msg)
            # If tool message, it's part of reasoning
            elif msg.type == "tool":
                reasoning_msgs.append(msg)
            # Last AI message without tool calls is final output
            elif msg.type == "ai" and i == len(new_messages) - 1:
                final_output = msg
            # Other AI messages in the middle might be intermediate thoughts
            elif msg.type == "ai":
                reasoning_msgs.append(msg)

        # Display reasoning process
        if reasoning_msgs:
            print("\n" + "=" * 80)
            print("Thinking:")
            print("=" * 80 + "\n")
            for msg in reasoning_msgs:
                if hasattr(msg, 'content') and msg.content:
                    if msg.type == "ai":
                        print(f"\n{msg.content}\n")
                    elif msg.type == "tool":
                        print(f"OBSERVATION: {msg.content}\n")

                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        print(f"ACTION: {tool_call['name']}({tool_call['args']})\n")

        # Display final output separator
        if final_output and final_output.content:
            print("=" * 80)
            print("FINAL OUTPUT:")
            print("=" * 80)
            print(f"\n{final_output.content}\n")
            print("=" * 80)

        print("\n")
