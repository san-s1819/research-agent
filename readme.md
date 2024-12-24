 # AI Research Assistant 🤖📚

An intelligent research assistant that reads, understands, and synthesizes information from academic papers using LangChain, Vector Search, and LLMs.

[Read the full tutorial on Medium](https://medium.com/@iamsanjay98/building-an-ai-research-assistant-with-langchain-pinecone-and-gpt-4-0873a5b4d959) 
## Features ✨

- Semantic search through research papers
- Natural language querying
- Automated research synthesis
- Structured report generation
- Vector-based paper similarity matching

## Prerequisites 🛠️

- Python 3.8 or higher
- API keys for:
  - OpenAI 
  - Pinecone
  - SerpAPI

## Quick Start 🚀

1. **Clone the repository**

`git clone https://github.com/san-s1819/research-agent.git
cd research-agent`



3. **Set up environment**
   
`python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt`



3. **Create `.env` file using the .env.example file**

`OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
SERPAPI_KEY=your_serpapi_key_here`


4. **Run the application**

`python agent-4o.py`


5. **Run a research query**

`out = runnable.invoke({
"input": "What are the latest developments in large language models?",
"chat_history": []
})`

6. **Print the formatted report**

`print(build_report(
output=out["intermediate_steps"][-1].tool_input
))`

7. **Directory Structure**

ai-research-assistant/\
├── research-agent.py # Main agent logic\
├── requirements.txt # Project dependencies\
├── .env # Environment variables (create this)\
└── README.md # This file

8. **Output format**

 ![image](https://github.com/user-attachments/assets/378d3dd5-1ab9-409b-ad74-0b9c59b5701d)


Don't forget to star ⭐ this repository if you found it helpful!
