from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langsmith import traceable

MAX_ITERATIONS=4
MODEL="qwen3:1.7b"

#------Tools (lang chain decorator)--------

@tool
def get_product_price(product:str) -> float:
    """Look up the product price in the catalog"""
    print(f"   >> Executing get_product_price (product='{product}') ")
    prices = {"laptop":1299,"desktop":600,"headphones":300}
    return prices.get(product, 0)

@tool
def apply_discount(price:float,discount_tier:str) -> float:
    """Apply discount to price and get final price
    Available tiers bronze silver and gold"""
    print(f" >> Executing apply_discount(price = {price}, discount = '{discount_tier}')")
    discount_percentages = {"bronze":4.0,"silver":8.0,"gold":12.0}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1 - discount /100),2)


@traceable(name = "LangChain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price,apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("="*60)

    messages = [SystemMessage(content=("You are a helpful shopping assistant"
                                      "You have access to product catalog tool and discount tool"
                                      "\n\nSTRICT RULES: you must follow these exactly"
                                      " 1. NEVER guess or assume any product price."
                                      "You must call get_product_price first to get real price"
                                      "2. Only call apply_discount only after you have received price from"
                                      "get_product_price. Pass the exact price returned by get_product_price"
                                      "DO NOT pass a made-up number\n"
                                      "3.NEVER Calculate discount yourself using math. Always use apply_discount tool\n "
                                      "4. If user does not specify discount tier, ask them with tier to use  - DO NOT assume one")
                                ),
                HumanMessage(content=question)]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"Iteration:  {iteration} ---")
        ai_message = llm_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        #If no toolcalls , then its the final answer
        if not tool_calls:
            print(f"Final answer --> {ai_message.content}")
            return ai_message.content

        #Process only first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"Tool selected : {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"{tool_name} not found ")
        observation = tool_to_use.invoke(tool_args)
        print(f"Tool result :  {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation),tool_call_id=tool_call_id))

    print(f"ERROR: Max iterations reached without final answer")
    return None

if __name__ == "__main__":
    print("hello langchain agent (.bind_tools)! ")
    print()
    result = run_agent("what is the price of laptop after applying gold discount?")

