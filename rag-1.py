from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, SecretStr
import os

load_dotenv()

class Summary(BaseModel):
    summary:str = Field('Short Summary of the data provided')

loader = TextLoader('data/text/cricket.txt', encoding='utf8')

docs = loader.load()

parser = PydanticOutputParser(pydantic_object=Summary)

model = ChatGroq(api_key=SecretStr(os.getenv('GROQ_API_KEY','')), model='llama-3.3-70b-versatile')

template = PromptTemplate(
    template='summarise the given topic : {topic}\n{format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain = template | model | parser

res=chain.invoke({'topic':docs})

print(res.summary)