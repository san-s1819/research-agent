from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import time
import os

dataset = load_dataset("jamescalam/ai-arxiv2-semantic-chunks", split="train")
#print(dataset[0])


encoder = SentenceTransformer('all-MiniLM-L6-v2')

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") 
# configure client
pc = Pinecone(api_key=api_key)


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


