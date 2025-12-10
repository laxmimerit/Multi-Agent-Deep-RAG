"""Yahoo Finance MCP module with LangChain integration."""
######## MCP SETUP ###############
# MCP GITHUB
# https://github.com/laxmimerit/MCP-Mastery-with-Claude-and-Langchain
# https://github.com/laxmimerit/Agentic-RAG-with-LangGraph-and-Ollama

# https://github.com/langchain-ai/langchain-mcp-adapters

import warnings 
warnings.filterwarnings("ignore")

import os
import sys

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

system_prompt = """
                You are a financial research assistant helping users analyze stocks and financial data using Yahoo Finance.

                Available Tools:
                - get_historical_stock_prices: Get historical stock prices (ticker required, optional: period='1mo', interval='1d')
                - get_stock_info: Get comprehensive stock information including price, metrics, financials, etc.
                - get_yahoo_finance_news: Get latest news for a stock ticker
                - get_stock_actions: Get dividends and stock splits information
                - get_financial_statement: Get financial statements (ticker and financial_type required: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow)
                - get_holder_info: Get holder information (ticker and holder_type required: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders)
                - get_option_expiration_dates: Get available option expiration dates
                - get_option_chain: Get option chain data (ticker, expiration_date, option_type required: 'calls' or 'puts')
                - get_recommendations: Get analyst recommendations (ticker and recommendation_type required: recommendations, upgrades_downgrades, optional: months_back=12)

                Instructions:
                - ALWAYS start by calling relevant tools to gather financial data when user asks about stocks
                - Extract ticker symbol from user query (e.g., AAPL, MSFT, GOOGL)
                - For general stock inquiries, start with get_stock_info to get comprehensive data
                - For price analysis, use get_historical_stock_prices with appropriate period
                - For news and sentiment, use get_yahoo_finance_news
                - Present data in a clear, organized format with key insights highlighted
                - Include specific numbers, percentages, and trends in your analysis
                - Be proactive - gather data first, then provide comprehensive analysis
                """

async def get_tools():
    client = MultiServerMCPClient(
        {
            "yfinance": {
                "command": "uvx",
                "args": ["yahoo-finance-mcp-server"],
                "transport": "stdio"
            }
        }
    )

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} Tools")
    print(f"Tools Available: {[tool.name for tool in tools]}")

    return tools

async def finance_research(query):
    tools = await get_tools()
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    result = await agent.ainvoke({'messages': [HumanMessage(query)]})

    response = result['messages'][-1].content[0]['text']
    print(response)

    return response

if __name__=="__main__":
    query = "What is the current stock price and recent performance of Apple (AAPL)? Also show me the latest news."
    asyncio.run(finance_research(query))
