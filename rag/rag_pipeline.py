"""
RAG Pipeline - Retrieve and Generate answers with citations
"""

from typing import Dict, List, Optional, Tuple
from rag.retriever import ChatRetriever
from llm.gemini_client import GeminiClient
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
        self.llm = GeminiClient()
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
            retrieved_messages = self.retriever.retrieve_by_query(
                query,
                top_k=top_k,
                date_range=date_range,
                include_context=True,
                context_size=self.context_window
            )
            
            if not retrieved_messages:
                return {
                    "answer": "I couldn't find relevant information in the specified date range.",
                    "citations": [],
                    "confidence": 0.0,
                    "sources_count": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Step 1.5: Add surrounding messages for better context (conversation flow)
            # Include 10 messages before and after every retrieved hit
            messages_with_context = []
            for msg in retrieved_messages:
                context_msgs = msg.get('context_messages')
                if not context_msgs:
                    context_msgs = self.retriever.get_context_around_message(
                        msg.get('message_id', ''),
                        context_size=self.context_window
                    )
                if not context_msgs:
                    context_msgs = [msg]
                messages_with_context.extend(context_msgs)
            
            # Deduplicate while preserving order
            seen_ids = set()
            unique_messages = []
            for msg in messages_with_context:
                msg_id = msg.get('message_id', '')
                if msg_id not in seen_ids:
                    seen_ids.add(msg_id)
                    unique_messages.append(msg)
            
            # Use messages with context if available, otherwise use retrieved messages
            final_messages = unique_messages if unique_messages else retrieved_messages
            
            # Step 2: Generate prompt with context
            prompt = get_qa_prompt(query, final_messages)
            
            # Step 3: Generate answer
            answer = self.llm.generate_text(prompt)
            
            # Ensure answer is a string
            if not answer:
                answer = "Unable to generate answer from the context provided."
            answer = str(answer).strip()
            
            # Step 4: Format citations
            citations = []
            for msg in retrieved_messages:
                try:
                    # Safely extract message content with fallbacks
                    message_text = msg.get('message')
                    if not message_text:
                        message_text = msg.get('original_message', 'N/A')
                    
                    citation = {
                        "text": str(message_text) if message_text else 'N/A',
                        "user": msg.get('user', 'Unknown'),
                        "timestamp": msg.get('date', msg.get('timestamp', 'N/A')),
                        "message_id": msg.get('message_id', ''),
                        "similarity_score": msg.get('similarity_score', 0.0)
                    }
                    citations.append(citation)
                except Exception as cite_err:
                    print(f"[DEBUG] Error processing citation: {str(cite_err)}")
                    continue
            
            return {
                "answer": answer,
                "citations": citations,
                "confidence": sum(msg.get('similarity_score', 0.0) for msg in retrieved_messages) / len(retrieved_messages) if retrieved_messages else 0.0,
                "sources_count": len(retrieved_messages),
                "processing_time": time.time() - start_time
            }
        
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"[DEBUG] RAG Pipeline Error: {error_msg}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            
            return {
                "answer": f"An error occurred: {error_msg[:100]}. Please try again or check your data.",
                "citations": [],
                "confidence": 0.0,
                "sources_count": 0,
                "processing_time": time.time() - start_time,
                "error_details": traceback.format_exc()
            }