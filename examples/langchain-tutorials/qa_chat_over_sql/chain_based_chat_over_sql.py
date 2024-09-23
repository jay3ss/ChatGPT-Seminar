from operator import itemgetter

from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

answer_prompt = PromptTemplate.from_template(
    """Given the following user quesiton, corresponding SQL Query, and SQL result, answer the question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

llm = ChatOpenAI(model="gpt-4o-mini")
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
write_query = create_sql_query_chain(llm, db)
filter_query = RunnableLambda(lambda x: x.replace("SQLQuery: ", "")) | RunnableLambda(
    lambda x: x.replace("```: ", "").replace("sql", "")
)
execute_query = QuerySQLDataBaseTool(db=db)
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | filter_query | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke({"question": "How many artists are there?"})
print(response)
