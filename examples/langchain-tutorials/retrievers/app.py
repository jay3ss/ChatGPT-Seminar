from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# define sample documents
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# initialize vector store with Chroma and OpenAI embeddings
vectorstore = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())

# define retriever using similarity search method
retreiver = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

# create a retrieveer from the vector store
vectorstore_retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 1}
)
# define the context and question for the RAG chain
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

# create the prompt template
prompt = ChatPromptTemplate.from_messages([("human", message)])

# initialize the LLM (ChatOpenAI) with the specified model
llm = ChatOpenAI(model="gpt-4o-mini")

# define the RAG chain
rag_chain = (
    {"context": vectorstore_retriever, "question": RunnablePassthrough()} | prompt | llm
)


# function to run the RAG chain and print the response
def run_rag_chain(question: str) -> None:
    response = rag_chain.invoke(question)
    print(response.content)


# example usage
if __name__ == "__main__":
    question = "tell me about cats"
    run_rag_chain(question)
