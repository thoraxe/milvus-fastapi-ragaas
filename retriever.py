# https://docs.llamaindex.ai/en/stable/examples/vector_stores/MilvusIndexDemo.html

import os
from dotenv import load_dotenv

# document indexing and embedding

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore

load_dotenv()

llm = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["OPENAI_API_KEY"],
    api_version="2023-03-15-preview",
    engine="0301-dep",
    model="gpt-3.5-turbo",
    temperature=0.3,
)

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

vector_store = MilvusVectorStore(
    uri="http://localhost:19530", dim=384, collection_name="openshift"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
vector_retriever = VectorIndexRetriever(index=index)

# fastapi implementation
from fastapi import FastAPI

from pydantic import BaseModel


class DocumentRequest(BaseModel):
    query: str


app = FastAPI()

@app.post("/query")
def user_query(document_request: DocumentRequest):
  nodes = vector_retriever.retrieve(document_request.query)
  return {"nodes": nodes}
