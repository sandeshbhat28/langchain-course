import os
from re import search
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
#from tavily import TavilyClient
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    """schema for a source used by agent"""

    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """"schema for the agent response with answer and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(description = "sources used to get answer")


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

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("hello from langchain course")
    result = agent.invoke({"messages": [HumanMessage(
        content="i want to search for 3 top sap specific tosca job openings in usa posted in last 4 hours on linkedin and list their details ?")]})
    print(result)


if __name__ == "__main__":
    main()
