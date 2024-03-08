import os
import jinja2
import requests

from dotenv import load_dotenv

from fastapi import FastAPI

from pydantic import BaseModel

load_dotenv()

from langchain_openai import AzureOpenAI


jinja_prompt_template_source = """
Instructions:
- You are a helpful assistant.
- You are an expert in Kubernetes and OpenShift.
- Respond to questions about topics other than Kubernetes and OpenShift with: "I can only answer questions about Kubernetes and OpenShift"
- Refuse to participate in anything that could harm a human.
- Base your answer on the provided context and query and not on prior knowledge.
- Use the following documentation sections as context

{% for node in nodes %}
  {{ node["node"]["metadata"]["file_path"] }}
  {{ node["node"]["text"] }}

{% endfor %}

Please answer the following question:
{{ query }}
"""

unrendered_jinja_prompt_template = jinja2.Environment().from_string(jinja_prompt_template_source)

llm = AzureOpenAI(deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"])

class UserQuery(BaseModel):
    query: str


app = FastAPI() 

@app.post("/query")
def user_query(user_query: UserQuery):
  # call the retriever to fetch nodes (llamaindex terminology) based on the query
  r = requests.post(f"http://{os.environ["RETRIEVER_SERVICE_NAME"]}:8000/query", json={"query": user_query.query})

  # TODO: error handling
  nodes = r.json()["nodes"]

  #embed()
  final_query = unrendered_jinja_prompt_template.render(nodes=nodes, query=user_query.query)

  return llm.invoke(final_query)
