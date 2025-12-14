from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
import yfinance as yf
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

load_dotenv()


def get_stock_prices(ticker: str, start_year: int = None, end_year: int = None) -> str:
    """
    Fetches historical daily closing prices for a given stock ticker
    between two years. Saves the results as a CSV file.

    Args:
        ticker (str): Stock ticker symbol.
        start_year (int): Starting year (e.g., 2022).
        end_year (int): Ending year (e.g., 2025).

    Returns:
        str: Path of the saved CSV file.
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

    # Keep only Close column
    df = df[["Close"]]

    # Save file
    filename = f"{ticker}_{start_year}_to_{end_year}.csv"
    df.to_csv(filename)

    return {"message": f"Saved prices to {filename}", "csv_file": filename}


def plot_stock_prices(csv_file: str) -> str:
    """
    Plots the time-series contained in a CSV file with a Close column.
    Saves plot as plot.png.
    """
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    # Derive plot filename from csv filename
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    plot_file = f"{base_name}_plot.png"

    plt.figure(figsize=(10, 4))
    plt.plot(df["Date"], df["Close"])
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"Closing Prices from {base_name}")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_file)
    plt.close()
    return f"Plot saved to {plot_file}"


search = DuckDuckGoSearchRun()

tools = [search, get_stock_prices, plot_stock_prices]
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)  # swap with ChatGoogleGenerativeAI(...) to use Gemini
llm_with_tools = llm.bind_tools(tools)


# Node
def reasoner(state: MessagesState):
    sys_msg = SystemMessage(
        content=(
            "You are a financial analysis assistant with access to tools for fetching stock price data "
            "and plotting charts.\n\n"
            "When the user asks for stock price data for a date range or year range, you should:\n"
            "1. First call the `get_stock_prices` tool to retrieve the data.\n"
            "2. After receiving the CSV file path returned by that tool, you MUST call the "
            "`plot_stock_prices` tool to visualize the data.\n\n"
            "Always follow this chain: fetch → plot.\n\n"
            "Use ReAct style reasoning:\n"
            "Thought: describe what you need to do\n"
            "Action: call the correct tool\n"
            "Observation: interpret its output\n"
            "Thought: decide next action\n"
            "Action: call the next tool\n"
            "… and so on."
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


# Run
response = react_graph.invoke(
    {
        "messages": [
            HumanMessage(content="Show me google stock prices between 2024 and 2025.")
        ]
    }
)

for m in response["messages"]:
    m.pretty_print()
