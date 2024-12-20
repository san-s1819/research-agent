from typing import TypedDict, Annotated
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
import operator
import requests
from langchain_core.tools import tool
import re
from serpapi import GoogleSearch
import os
from getpass import getpass

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import time
import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor
from langchain.agents import create_structured_chat_agent

dataset = load_dataset("jamescalam/ai-arxiv2-semantic-chunks", split="train")
#print(dataset[0])


encoder = SentenceTransformer('all-MiniLM-L6-v2')

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone_api_key = os.getenv("PINECONE_API_KEY") 
# configure client
pc = Pinecone(api_key=pinecone_api_key)


from pinecone import ServerlessSpec

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"  # us-east-1
)
dims = len(encoder.encode(["some random text"])[0])

index_name = "llama-3-research-agent"

#check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=dims,  # dimensionality of embed 3
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
print(index.describe_index_stats())

from tqdm.auto import tqdm

# easier to work with dataset as pandas dataframe
data = dataset.to_pandas().iloc[:10000]

batch_size = 128

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i+batch_size)
    batch = data[i:i_end].to_dict(orient="records")
    # get batch of data
    metadata = [{
        "title": r["title"],
        "content": r["content"],
        "arxiv_id": r["arxiv_id"],
        "references": r["references"].tolist()
    } for r in batch]
    # generate unique ids for each chunk
    ids = [r["id"] for r in batch]
    # get text content to embed
    content = [r["content"] for r in batch]
    # embed text
    embeds = encoder.encode(content)
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))



class AgentState(TypedDict):
   input: str
   chat_history: list[BaseMessage]
   intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]




@tool("fetch_arxiv")
def fetch_arxiv(arxiv_id: str):
   """Gets the abstract from an ArXiv paper given the arxiv ID. Useful for
   finding high-level context about a specific paper."""
   # get paper page in html
   res = requests.get(
       f"https://export.arxiv.org/abs/{arxiv_id}"
   )
   abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
   )
   # search html for abstract
   re_match = abstract_pattern.search(res.text)
   # return abstract text
   return re_match.group(1)

# print(
#     fetch_arxiv.invoke(input={"arxiv_id": "2401.04088"})
# )



serpapi_params = {
    "engine": "google",
    "api_key": os.getenv("SERPAPI_KEY")
}

@tool("web_search")
def web_search(query: str):
   """Finds general knowledge information using Google search. Can also be used
   to augment more 'general' knowledge to a previous specialist query."""
   search = GoogleSearch({
       **serpapi_params,
       "q": query,
       "num": 5
   })
   results = search.get_dict()["organic_results"]
   contexts = "\n---\n".join(
       ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
   )
   return contexts

def format_rag_contexts(matches: list):
   contexts = []
   for x in matches:
       text = (
           f"Title: {x['metadata']['title']}\n"
           f"Content: {x['metadata']['content']}\n"
           f"ArXiv ID: {x['metadata']['arxiv_id']}\n"
           f"Related Papers: {x['metadata']['references']}\n"
       )
       contexts.append(text)
   context_str = "\n---\n".join(contexts)
   return context_str


@tool("rag_search_filter")
def rag_search_filter(query: str, arxiv_id: str):
   """Finds information from our ArXiv database using a natural language query
   and a specific ArXiv ID. Allows us to learn more details about a specific paper."""
   xq = encoder([query])
   xc = index.query(vector=xq, top_k=6, include_metadata=True, filter={"arxiv_id": arxiv_id})
   context_str = format_rag_contexts(xc["matches"])
   return context_str


@tool("rag_search")
def rag_search(query: str):
   """Finds specialist information on AI using a natural language query."""
   xq = encoder([query])
   xc = index.query(vector=xq, top_k=2, include_metadata=True)
   context_str = format_rag_contexts(xc["matches"])
   return context_str

@tool("final_answer")
def final_answer(
   introduction: str,
   research_steps: str,
   main_body: str,
   conclusion: str,
   sources: str
):
   """Returns a natural language response to the user in the form of a research
   report. There are several sections to this report, those are:
   - `introduction`: a short paragraph introducing the user's question and the
   topic we are researching.
   - `research_steps`: a few bullet points explaining the steps that were taken
   to research your report.
   - `main_body`: this is where the bulk of high quality and concise
   information that answers the user's question belongs. It is 3-4 paragraphs
   long in length.
   - `conclusion`: this is a short single paragraph conclusion providing a
   concise but sophisticated view on what was found.
   - `sources`: a bulletpoint list provided detailed sources for all information
   referenced during the research process
   """
   if type(research_steps) is list:
       research_steps = "\n".join([f"- {r}" for r in research_steps])
   if type(sources) is list:
       sources = "\n".join([f"- {s}" for s in sources])
   return ""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


system_prompt = """You are the oracle, the great AI decision maker.
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

Previous actions and results:
{scratchpad}

If you see that a tool has been used with a particular query, do NOT use that 
same tool with the same query again."""


prompt = ChatPromptTemplate.from_messages([
   ("system", system_prompt),
   MessagesPlaceholder(variable_name="chat_history"),
   ("user", "{input}"),
   ("assistant", "scratchpad: {scratchpad}"),
])

from langchain_core.messages import ToolCall, ToolMessage
from langchain_openai import ChatOpenAI


llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model_name="mixtral-8x7b-32768",  # or "llama2-70b-4096"
    temperature=0
)

tools=[
   rag_search_filter,
   rag_search,
   fetch_arxiv,
   web_search,
   final_answer
]


# define a function to transform intermediate_steps from list
# of AgentAction to scratchpad string
def create_scratchpad(intermediate_steps: list[AgentAction]):
   research_steps = []
   for i, action in enumerate(intermediate_steps):
       if action.log != "TBD":
           # this was the ToolExecution
           research_steps.append(
               f"Tool: {action.tool}, input: {action.tool_input}\n"
               f"Output: {action.log}"
           )
   return "\n---\n".join(research_steps)


# Create the agent with scratchpad-aware prompt
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt.partial(scratchpad=lambda x: create_scratchpad(x["intermediate_steps"]))
)

oracle = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

inputs = {
    "input": "tell me something interesting about dogs",
    "chat_history": [],
    "intermediate_steps": [],
}
out = oracle.invoke(inputs)
print(out.tool_calls[0]["name"])
print(out.tool_calls[0]["args"])

from typing import TypedDict


def run_oracle(state: dict):
   print("run_oracle")
   print(f"intermediate_steps: {state['intermediate_steps']}")
   out = oracle.invoke(state)
   tool_name = out.tool_calls[0]["name"]
   tool_args = out.tool_calls[0]["args"]
   action_out = AgentAction(
       tool=tool_name,
       tool_input=tool_args,
       log="TBD"
   )
   return {
       "intermediate_steps": [action_out]
   }


def router(state: dict):
   # return the tool name to use
   if isinstance(state["intermediate_steps"], list):
       return state["intermediate_steps"][-1].tool
   else:
       # if we output bad format go to final answer
       print("Router invalid format")
       return "final_answer"
