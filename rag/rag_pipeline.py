"""
RAG Pipeline - Retrieve and Generate answers with citations
WITH IMPROVED ERROR HANDLING FOR SAFETY BLOCKS
"""

from typing import Dict, List, Optional, Tuple
from rag.retriever import ChatRetriever
from llm.mistral_client import MistralClient
from llm.prompt_templates import get_qa_prompt
import time

class RAGPipeline: 
    """Orchestrate retrieval and generation"""
    
    def __init__(self, index_name: Optional[str] = None):
        """Initialize RAG pipeline

        Args:
            index_name: optional chat-specific FAISS index namespace
        """
        self.retriever = ChatRetriever(index_name=index_name)
        self.llm = MistralClient()
        self.context_window = 10
    
    def answer_question(
        self,
        query: str,
        date_range: Optional[Tuple[str, str]] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Answer question using RAG
        
        Args:
            query: user question
            date_range: optional (start_date, end_date) tuple
            top_k: number of messages to retrieve
        
        Returns:
            Dict with answer and citations
        """
        start_time = time.time()
        
        try:
            # Validate query
            if not query or not query.strip():
                return {
                    "answer": "Please ask a valid question.",
                    "citations": [],
                    "confidence": 0.0,
                    "sources_count": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Step 1: Retrieve relevant messages
            retrieved_messages = self.retriever. retrieve_by_query(
                query,
                top_k=top_k,
                date_range=date_range,
                include_context=True,
                context_size=self.context_window
            )
            
            if not retrieved_messages:
                return {
                    "answer": "I couldn't find relevant information in the specified date range.",
                    "citations":  [],
                    "confidence": 0.0,
                    "sources_count": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Step 1. 5: Add surrounding messages for better context
            messages_with_context = []
            for msg in retrieved_messages:
                context_msgs = msg.get('context_messages')
                if not context_msgs:
                    context_msgs = self.retriever.get_context_around_message(
                        msg. get('message_id', ''),
                        context_size=self.context_window
                    )
                if not context_msgs:
                    context_msgs = [msg]
                messages_with_context. extend(context_msgs)
            
            # Deduplicate while preserving order
            seen_ids = set()
            unique_messages = []
            for msg in messages_with_context:
                msg_id = msg.get('message_id', '')
                if msg_id not in seen_ids: 
                    seen_ids.add(msg_id)
                    unique_messages.append(msg)
            
            # Use messages with context if available
            final_messages = unique_messages if unique_messages else retrieved_messages
            
            # Step 2: Generate prompt with context
            prompt = get_qa_prompt(query, final_messages)
            
            # Step 3: Generate answer
            answer = self.llm.generate_text(prompt)
            
            # Check if response was blocked or errored
            if answer.startswith("[Error"):
                return {
                    "answer": "Unable to generate answer due to an error. Please try again.",
                    "citations": self._format_citations(retrieved_messages),
                    "confidence": 0.5,
                    "sources_count": len(retrieved_messages),
                    "processing_time": time.time() - start_time
                }
            
            # Step 4: Format response
            return {
                "answer": answer,
                "citations": self._format_citations(retrieved_messages),
                "confidence": self._estimate_confidence(retrieved_messages),
                "sources_count": len(retrieved_messages),
                "processing_time": time.time() - start_time
            }
        
        except Exception as e: 
            print(f"[RAG Pipeline Error] {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "citations":  [],
                "confidence": 0.0,
                "sources_count": 0,
                "processing_time": time.time() - start_time
            }
    
    def _format_citations(self, messages: List[Dict]) -> List[Dict]:
        """Format retrieved messages as citations"""
        citations = []
        for i, msg in enumerate(messages[: 10], 1):  # Limit to top 10
            citations.append({
                "id": i,
                "user":  msg.get("user", "Unknown"),
                "message": msg.get("message", ""),
                "timestamp": msg. get("date", ""),
                "score": round(msg.get("score", 0.0), 3)
            })
        return citations
    
    def _estimate_confidence(self, messages: List[Dict]) -> float:
        """Estimate confidence based on retrieval scores"""
        if not messages: 
            return 0.0
        
        # Average of top 3 scores
        top_scores = [msg.get("score", 0.0) for msg in messages[: 3]]
        if not top_scores:
            return 0.5
        
        avg_score = sum(top_scores) / len(top_scores)
        # Normalize to 0-1 range (scores are typically 0-2 for cosine similarity)
        confidence = min(avg_score / 2.0, 1.0)
        return round(confidence, 2)
