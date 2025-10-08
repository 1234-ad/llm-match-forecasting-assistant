"""
RAG Implementation for Match Data Retrieval
Handles vector storage and semantic search of match statistics
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb
from chromadb.config import Settings


class MatchDataRetriever:
    """RAG-based retriever for match and team statistics"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            client=self.client,
            collection_name="match_data",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # Load initial data if empty
        if self.vectorstore._collection.count() == 0:
            self._load_initial_data()
    
    def _load_initial_data(self):
        """Load initial match data into vector store"""
        # Sample match data - in production, load from APIs/databases
        sample_data = [
            {
                "team": "Manchester United",
                "opponent": "Arsenal",
                "date": "2024-01-15",
                "result": "2-1",
                "venue": "Old Trafford",
                "stats": {
                    "possession": 58,
                    "shots": 14,
                    "shots_on_target": 6,
                    "corners": 7,
                    "fouls": 12
                },
                "form": "WWDLW",
                "injuries": ["Marcus Rashford", "Luke Shaw"],
                "weather": "Clear, 15°C"
            },
            {
                "team": "Liverpool",
                "opponent": "Chelsea",
                "date": "2024-01-20",
                "result": "1-1",
                "venue": "Anfield",
                "stats": {
                    "possession": 62,
                    "shots": 18,
                    "shots_on_target": 8,
                    "corners": 9,
                    "fouls": 8
                },
                "form": "WWWDL",
                "injuries": ["Virgil van Dijk"],
                "weather": "Rainy, 12°C"
            },
            {
                "team": "Manchester City",
                "opponent": "Tottenham",
                "date": "2024-01-25",
                "result": "3-0",
                "venue": "Etihad Stadium",
                "stats": {
                    "possession": 71,
                    "shots": 22,
                    "shots_on_target": 12,
                    "corners": 11,
                    "fouls": 6
                },
                "form": "WWWWW",
                "injuries": [],
                "weather": "Cloudy, 8°C"
            }
        ]
        
        documents = []
        for match in sample_data:
            content = self._format_match_data(match)
            doc = Document(
                page_content=content,
                metadata={
                    "team": match["team"],
                    "opponent": match["opponent"],
                    "date": match["date"],
                    "result": match["result"],
                    "venue": match["venue"]
                }
            )
            documents.append(doc)
        
        # Add documents to vector store
        self.vectorstore.add_documents(documents)
        print(f"Loaded {len(documents)} match records into vector store")
    
    def _format_match_data(self, match: Dict[str, Any]) -> str:
        """Format match data for vector storage"""
        stats_str = ", ".join([f"{k}: {v}" for k, v in match["stats"].items()])
        injuries_str = ", ".join(match["injuries"]) if match["injuries"] else "None"
        
        return f"""
        Match: {match['team']} vs {match['opponent']}
        Date: {match['date']}
        Result: {match['result']}
        Venue: {match['venue']}
        Form: {match['form']}
        Statistics: {stats_str}
        Injuries: {injuries_str}
        Weather: {match['weather']}
        """
    
    def retrieve_relevant_matches(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant match data based on query"""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"Error retrieving matches: {e}")
            return []
    
    def retrieve_team_stats(self, team_name: str, k: int = 3) -> List[Document]:
        """Retrieve specific team statistics"""
        query = f"team statistics for {team_name} recent matches form performance"
        return self.retrieve_relevant_matches(query, k)
    
    def retrieve_head_to_head(self, team1: str, team2: str, k: int = 3) -> List[Document]:
        """Retrieve head-to-head match history"""
        query = f"{team1} vs {team2} head to head historical matches"
        return self.retrieve_relevant_matches(query, k)
    
    def add_match_data(self, match_data: Dict[str, Any]):
        """Add new match data to the vector store"""
        content = self._format_match_data(match_data)
        doc = Document(
            page_content=content,
            metadata={
                "team": match_data["team"],
                "opponent": match_data["opponent"],
                "date": match_data["date"],
                "result": match_data.get("result", ""),
                "venue": match_data.get("venue", "")
            }
        )
        self.vectorstore.add_documents([doc])
    
    def as_tool(self) -> Tool:
        """Convert retriever to LangChain tool"""
        def retrieve_match_data(query: str) -> str:
            """Retrieve relevant match data and statistics"""
            docs = self.retrieve_relevant_matches(query, k=5)
            if not docs:
                return "No relevant match data found."
            
            result = "Relevant match data:\n\n"
            for i, doc in enumerate(docs, 1):
                result += f"{i}. {doc.page_content}\n"
                result += f"   Metadata: {doc.metadata}\n\n"
            
            return result
        
        return Tool(
            name="retrieve_match_data",
            description="Retrieve historical match data, team statistics, and relevant context for predictions",
            func=retrieve_match_data
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        try:
            count = self.vectorstore._collection.count()
            return {
                "total_documents": count,
                "collection_name": "match_data",
                "embedding_model": "text-embedding-ada-002"
            }
        except Exception as e:
            return {"error": str(e)}