from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, SecretStr
import os

load_dotenv()

class UserQuery(BaseModel):
    answer:str = Field('Answer to the user query from givrn context')

loader=PyPDFLoader('data/pdf/Rishav_Chatterjee_FSD_Resume_2025.pdf')

docs = loader.load()

parser = PydanticOutputParser(pydantic_object=UserQuery)

model = ChatGroq(api_key=SecretStr(os.getenv('GROQ_API_KEY','')), model='llama-3.3-70b-versatile')

template = PromptTemplate(
    template='resolve user query : {query} on the the given context : {context}\n{format_instructions}',
    input_variables=['query','context'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain = template | model | parser

res=chain.invoke({'context':docs,'query':'What is the overall years of experience of the user?'})

print(res.answer)

