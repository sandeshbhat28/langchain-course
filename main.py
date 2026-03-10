import os
from re import search

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
#from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

# Tavily Client used if we want to write our tool decorator function.
# Else go with inbuilt TavilySearch
#tavily = TavilyClient()

# @tool
# def search(query: str) -> dict:
#     """tool that searches over internet
#     Args:
#         query :  query to search for
#     Returns : search result """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)


#llm = ChatOllama(model="llama3.1:latest")
#llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

llm = ChatOllama(model="mistral-large-3:675b-cloud")

tools = [TavilySearch()]

# Tavily Search used like above if we want Tavily to manage
# else go with own tools capabilities
#tools = [search]

agent = create_agent(model=llm, tools=tools)


def main():
    print("hello from langchain course")
    result = agent.invoke({"messages": [HumanMessage(
        content="i want to search for 3 top tosca job openings in usa oregon(portland, beaverton, hillsboro)  on linkedin and list their details ?")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
