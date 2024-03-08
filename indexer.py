# https://docs.llamaindex.ai/en/stable/examples/vector_stores/MilvusIndexDemo.html

import os

from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

documents = SimpleDirectoryReader("./ocp-product-docs-plaintext", recursive=True).load_data()

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
    uri="http://localhost:19530", dim=384, overwrite=True, collection_name="openshift"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# store the vectors
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

print("Done indexing!")