import os
import requests
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json


load_dotenv()

@tool
def get_conversion_factor(base_currency: str, target_currency:str)->float:
    """
    This function fetches the currency conversionnfactor between a given base currency and a target currency
    """
    url=f"https://v6.exchangerate-api.com/v6/{os.getenv('CURRENCY_API_KEY','')}/pair/{base_currency}/{target_currency}"

    response = requests.get(url)

    return response.json()

@tool
def convert(base_currency_value: int, convertion_rate:Annotated[float, InjectedToolArg]) -> float:
    """
    given a currency conversion rate -> This function calculates the target currency value from a given base currency value
    """

    return base_currency_value * convertion_rate

llm = ChatGroq(api_key=SecretStr(os.getenv('GROQ_API_KEY','')), model='llama-3.3-70b-versatile')

llm_with_tools = llm.bind_tools([get_conversion_factor,convert])

messages=[HumanMessage(content="What is the conversion factor between USD and INR and based on that can you convert 10 USD to INR?")]

ai_message = llm_with_tools.invoke(messages) #gives back the tools and args

messages.append(ai_message)

conversion_rate=0

for tool_call in ai_message.tool_calls: #type:ignore
    if tool_call['name'] == 'get_conversion_factor':
        tool_msg1=get_conversion_factor.invoke(tool_call)
        conversion_rate=json.loads(tool_msg1.content)['conversion_rate']
        messages.append(tool_msg1)

    elif tool_call['name'] == 'convert':
        tool_call['args']['convertion_rate']=conversion_rate
        tool_msg2 = convert.invoke(tool_call)
        messages.append(tool_msg2)

print(llm_with_tools.invoke(messages).content)

