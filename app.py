#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG Chatbot for Financial Statements

This file implements an end-to-end Retrieval-Augmented Generation (RAG) Chatbot that answers
financial queries based on a company's financial statements (last two years).

The tasks included are:
1. Data Collection & Preprocessing:
   - Loads a financial report and splits the report into overlapping text chunks (Chunk Merging).
2. Basic RAG Implementation:
   - Uses an embedding model (all-MiniLM-L6-v2 from SentenceTransformers) to encode text chunks.
3. Advanced RAG Implementation:
   - Combines BM25 keyword-based retrieval (using rank_bm25) with embedding-based retrieval 
     (adaptive retrieval) by merging and re-ranking scores.
4. Guardrail Implementation:
   - Validates that user queries are financially related by checking for key financial keywords.
5. Application Interface using Streamlit:
   - A web-based interface that maintains conversation history, allows multi-turn interaction, 
     displays a confidence score (derived from cosine similarity) with each response, and shows a 
     loader while the query is being processed.
     
Usage:
    Place your financial report in "data/financial_report.txt"
    Run: streamlit run rag_chatbot.py
"""

import os
import streamlit as st
import numpy as np

# Importing necessary modules for embedding, BM25, and response generation
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from transformers import pipeline

# -------------------------------
# 1. Data Collection & Preprocessing
# -------------------------------

def load_financial_report(filepath: str) -> str:
    """
    Load financial report text file.

    Parameters:
        filepath (str): Path to the financial report file.
    
    Returns:
        str: Content of the file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def chunk_text(text: str, max_length: int = 300, overlap: int = 50) -> list:
    """
    Chunk text into pieces with a maximum number of words and with an overlap between chunks.
    
    This overlapping technique (Chunk Merging) is used to preserve context across text segments.
    
    Parameters:
        text (str): Input text.
        max_length (int): Maximum number of words in a chunk.
        overlap (int): Number of words that overlap between consecutive chunks.
    
    Returns:
        list: List of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_length
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        if end >= len(words):
            break
        start = end - overlap  # Overlap to include context in next chunk
    return chunks

# -------------------------------
# 2. Embedding-based Retrieval
# -------------------------------

class EmbeddingRetriever:
    """
    An embedding-based retriever that encodes text chunks and retrieves them based on cosine similarity.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model and storage for document embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.document_chunks = []
        self.embeddings = None
        
    def add_documents(self, chunks: list):
        """
        Add document chunks and precompute their embeddings.
        """
        self.document_chunks = chunks
        self.embeddings = self.model.encode(chunks, convert_to_tensor=True)
        
    def search(self, query: str, top_k: int = 5):
        """
        Search for the top_k most similar document chunks given a query.
        
        Parameters:
            query (str): The user's query.
            top_k (int): Number of top results to return.
        
        Returns:
            list: Each element is a tuple (chunk, cosine similarity score).
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        actual_top_k = min(top_k, cosine_scores.shape[0])
        top_results = np.argpartition(-cosine_scores.cpu().numpy(), range(actual_top_k))[0:actual_top_k]
        top_results = sorted(top_results, key=lambda idx: cosine_scores[idx].item(), reverse=True)
        return [(self.document_chunks[idx], cosine_scores[idx].item()) for idx in top_results]

# -------------------------------
# 3. BM25 Retrieval
# -------------------------------

class BM25Retriever:
    """
    A BM25 retriever using rank_bm25 for keyword-based document retrieval.
    """
    def __init__(self, chunks: list):
        """
        Initialize BM25 retriever with document chunks.
        """
        self.chunks = chunks
        self.corpus_tokens = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
    def search(self, query: str, top_k: int = 5):
        """
        Search for the top_k relevant document chunks using BM25.
        
        Parameters:
            query (str): The user query.
            top_k (int): Number of top results to return.
        
        Returns:
            list: Each element is a tuple (chunk, BM25 score).
        """
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        actual_top_k = min(top_k, len(self.chunks))
        if actual_top_k == 0:
            return []
        top_indices = np.argpartition(-scores, range(actual_top_k))[0:actual_top_k]
        top_indices = sorted(top_indices, key=lambda idx: scores[idx], reverse=True)
        return [(self.chunks[idx], scores[idx]) for idx in top_indices]

# -------------------------------
# 4. Response Generation
# -------------------------------

class ResponseGenerator:
    """
    A response generator that uses a language model to produce answers based on query and context.
    """
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize the response generation model.
        """
        self.generator = pipeline("text2text-generation", model=model_name)
        
    def generate_response(self, query: str, context: str, max_length: int = 150) -> str:
        """
        Generate a response by combining the query with the retrieved context.
        
        Parameters:
            query (str): The user's query.
            context (str): Context retrieved from the financial report.
            max_length (int): Maximum length of the generated answer.
        
        Returns:
            str: The generated answer.
        """
        prompt = f"Answer the financial query: {query}\nBased on the following context: {context}\nAnswer:"
        response = self.generator(prompt, max_length=max_length, do_sample=False)
        return response[0]['generated_text']

# -------------------------------
# 5. Guardrail Implementation
# -------------------------------

def validate_query(query: str) -> (bool, str):
    """
    Validate the query by checking for common financial keywords.
    
    Parameters:
        query (str): The user's query.
    
    Returns:
        tuple: (bool, str) where bool is True if the query is valid, and str is an error
        message if not.
    """
    financial_keywords = ["revenue", "profit", "earnings", "financial", "statement",
                          "balance sheet", "income", "cash flow", "quarter", "year", "growth"]
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in financial_keywords):
        return True, ""
    else:
        return False, "Query does not appear to be related to financial statements. Please ask questions about the company's financials."

# -------------------------------
# 6. Adaptive Retrieval: Combining BM25 and Embedding Scores
# -------------------------------

def combined_search(query: str, emb_retriever: EmbeddingRetriever, bm25_retriever: BM25Retriever, top_k: int = 5):
    """
    Combine the results from embedding-based and BM25-based retrieval methods.
    
    Parameters:
        query (str): The user's query.
        emb_retriever (EmbeddingRetriever): Instance of the embedding retriever.
        bm25_retriever (BM25Retriever): Instance of the BM25 retriever.
        top_k (int): Number of top results to consider from each method.
    
    Returns:
        list: Sorted list of (chunk, combined score) tuples.
    """
    emb_results = emb_retriever.search(query, top_k=top_k)
    bm25_results = bm25_retriever.search(query, top_k=top_k)
    
    scores = {}
    for chunk, score in emb_results:
        scores[chunk] = scores.get(chunk, 0) + score
        
    if bm25_results:
        max_bm25 = max([score for _, score in bm25_results])
        for chunk, score in bm25_results:
            normalized_score = score / max_bm25 if max_bm25 > 0 else 0
            scores[chunk] = scores.get(chunk, 0) + normalized_score
            
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_chunks

# -------------------------------
# 7. Chatbot UI with Streamlit (with Skeleton Loader)
# -------------------------------

def main():
    """
    The main function to run the RAG Chatbot using Streamlit.
    
    Functionality:
      1. Validates the user query using a guardrail.
      2. Loads and preprocesses the financial report.
      3. Initializes the retrievers and response generator.
      4. Retrieves relevant context via adaptive (combined) retrieval.
      5. Generates a response and computes a confidence score.
      6. Displays a skeleton loader (spinner) while the query is being processed.
      7. Maintains a conversation history for multi-turn interaction.
    """
    st.title("RAG Chatbot for Financial Statements")
    st.write("Ask questions about the company's financial reports (last two years).")
    
    # Initialize session state for conversation history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat submission form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Enter your financial query:")
        submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input:
        # Show a loader while processing the query
        with st.spinner("Processing query..."):
            # Validate query using guardrail
            is_valid, message = validate_query(user_input)
            if not is_valid:
                response = "Guardrail Alert: " + message
                avg_confidence = 0.0
            else:
                # Load and preprocess the financial report
                data_path = os.path.join("data", "financial_report.txt")
                report_text = load_financial_report(data_path)
                chunks = chunk_text(report_text, max_length=300, overlap=50)
                
                # Initialize retrievers and response generator
                emb_retriever = EmbeddingRetriever()
                emb_retriever.add_documents(chunks)
                bm25_retriever = BM25Retriever(chunks)
                response_generator = ResponseGenerator()
                
                # Retrieve context and generate a response (adaptive retrieval)
                results = combined_search(user_input, emb_retriever, bm25_retriever, top_k=5)
                top_chunks = [chunk for chunk, score in results[:3]]
                context = "\n".join(top_chunks)
                response = response_generator.generate_response(user_input, context)
                
                # Compute a dummy confidence score (average of top 3 cosine similarities)
                emb_scores = [score for _, score in emb_retriever.search(user_input, top_k=3)]
                avg_confidence = sum(emb_scores) / len(emb_scores) if emb_scores else 0.0
                
            # Append the current conversation to the chat history
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": response,
                "confidence": avg_confidence
            })
    
    # Display the conversation history
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown(f"**Confidence Score:** {chat['confidence']:.2f}")
    
if __name__ == "__main__":
    main() 
