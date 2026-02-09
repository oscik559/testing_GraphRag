import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import LLMInterface

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    print("Error: Missing environment variables.")
    exit(1)

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Define Embeddings
embedder = OllamaEmbeddings(model="nomic-embed-text-v2-moe:latest")

# Define Retriever
# We retrieval on StandardDocument nodes using their embeddings
retriever = VectorRetriever(
    driver,
    index_name="standard_embeddings",
    embedder=embedder,
    return_properties=["title", "standard_code"]
)

# Define LLM
# Note: neo4j-graphrag might not have a direct 'OllamaLLM' class in all versions, 
# but usually supports anything matching their interface or via langchain adapter.
# For simplicity, we can use the OpenAILLM class pointed at localhost if compatible,
# OR assume there is an OllamaLLM. Let's check imports.
# Actually, the safest bet is usually to wrap LangChain's ChatOllama if neo4j-graphrag supports LangChain LLMs.
# Checking docs: neo4j-graphrag often supports LangChain LLMs directly or via a wrapper.

from langchain_ollama import ChatOllama

class LangChainLLM(LLMInterface):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input: str, **kwargs):
        # response = self.llm.invoke(input)
        # return response
        # Simplify to direct call
        return self.llm.invoke(input)

    async def ainvoke(self, input: str, **kwargs):
        return await self.llm.ainvoke(input)

llm = LangChainLLM(ChatOllama(model="llama3.2:latest", temperature=0))


# Define GraphRAG
rag = GraphRAG(
    retriever=retriever,
    llm=llm
)

def query_graph(question: str):
    print(f"\nQuestion: {question}")
    try:
        response = rag.search(query_text=question)
        print(f"Answer: {response.answer}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing query: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        query_graph(q)
    else:
        print("Usage: python graphrag_app.py 'Your question here'")
