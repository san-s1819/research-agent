import asyncio
from typing import List, Dict
from langchain.llms import OpenAI
from scholarly import scholarly
import spacy

class ResearchAssistantAgent:
    def __init__(self, api_keys):
        self.llm = OpenAI(api_key=api_keys['openai'])
        self.scholarly_client = scholarly
        self.nlp = spacy.load("en_core_web_sm")
    
    async def generate_research_questions(self, domain: str) -> List[str]:
        """Generate innovative research questions in a specific domain"""
        prompt = f"Generate 3 novel, unexplored research questions in {domain}"
        questions = self.llm.generate(prompt)
        return [q.strip() for q in questions.split('\n')]
    
    async def search_academic_databases(self, query: str) -> List[Dict]:
        """Search academic databases and retrieve relevant papers"""
        search_results = self.scholarly_client.search_pubs(query)
        return [
            {
                'title': paper.bib['title'],
                'authors': paper.bib['author'],
                'abstract': paper.bib.get('abstract', '')
            } 
            for paper in search_results
        ]
    
    async def analyze_literature(self, papers: List[Dict]) -> Dict:
        """Perform comprehensive literature analysis"""
        analysis_prompt = f"""
        Analyze the following research papers and provide:
        1. Key themes
        2. Methodological approaches
        3. Identified research gaps
        
        Papers: {papers}
        """
        analysis = self.llm.generate(analysis_prompt)
        return {
            'themes': self._extract_themes(analysis),
            'gaps': self._identify_knowledge_gaps(analysis)
        }
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract key research themes using NLP"""
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    
    def _identify_knowledge_gaps(self, analysis: str) -> List[str]:
        """Identify potential research gaps"""
        gap_prompt = f"Extract potential research gaps from: {analysis}"
        gaps = self.llm.generate(gap_prompt)
        return [gap.strip() for gap in gaps.split('\n')]
    
    async def generate_research_proposal(self, research_question: str) -> str:
        """Generate a comprehensive research proposal"""
        proposal_prompt = f"""
        Create a detailed research proposal for: {research_question}
        
        Include:
        - Background and significance
        - Research objectives
        - Proposed methodology
        - Expected outcomes
        """
        return self.llm.generate(proposal_prompt)

async def main():
    agent = ResearchAssistantAgent(api_keys={
        'openai': 'your_openai_api_key',
        'scholarly': 'your_scholarly_api_key'
    })
    
    domain = "Artificial Intelligence in Healthcare"
    questions = await agent.generate_research_questions(domain)
    
    for question in questions:
        papers = await agent.search_academic_databases(question)
        literature_analysis = await agent.analyze_literature(papers)
        research_proposal = await agent.generate_research_proposal(question)
        
        print(f"Research Question: {question}")
        print(f"Literature Analysis: {literature_analysis}")
        print(f"Proposal: {research_proposal}")

if __name__ == "__main__":
    asyncio.run(main())