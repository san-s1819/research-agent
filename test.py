from serpapi import GoogleSearch
import os
from getpass import getpass
serpapi_params = {
    "engine": "google",
    "api_key": os.getenv("SERPAPI_KEY") 
}

search = GoogleSearch({
    **serpapi_params,
    "q": "coffee"
})

results = search.get_dict()["organic_results"]

contexts = "\n---\n".join(
    ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
)

print(contexts)


