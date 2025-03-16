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
     displays a confidence score with each response, and shows a loader while the query is being processed.
     
Usage:
    Place your financial report in "data/financial_report.txt"
    Run: streamlit run rag_chatbot.py
"""

import os
import streamlit as st
import numpy as np
import math

# Import necessary modules
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from transformers import pipeline

# --- NLTK Setup for Streamlit Cloud ---
# Set a custom download directory (e.g., /tmp/nltk_data) and add it to nltk.data.path so that NLTK can find the resources.
import nltk
nltk_data_dir = '/tmp/nltk_data'
nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

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

def chunk_text(text: str, max_length: int = 150, overlap: int = 75) -> list:
    """
    Chunk text into coherent pieces using sentence boundaries and overlapping sentences.
    
    This function implements the "Chunk Merging" strategy:
      - It splits the input text based on sentence tokenization using NLTK.
      - Each chunk is assembled to have up to `max_length` words.
      - To preserve context between chunks, the final sentences (accumulating at least `overlap` words)
        from a finished chunk are added to the beginning of the next chunk.
      
    This merging of overlapping chunks helps capture complete ideas and is crucial for retrieving
    well-contextualized information from the financial report.

    Parameters:
        text (str): The input text to be chunked.
        max_length (int): Maximum number of words per chunk (default: 150).
        overlap (int): Minimum number of words to include as overlap between consecutive chunks (default: 75).

    Returns:
        list: List of string chunks.
    """
    # Tokenize the text into sentences using NLTK's pre-downloaded 'punkt' data.
    sentences = nltk.tokenize.sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        # If adding this sentence does not exceed the maximum length, add it to the current chunk.
        if current_word_count + len(sentence_words) <= max_length:
            current_chunk.append(sentence)
            current_word_count += len(sentence_words)
        else:
            # Append the current chunk once the max length is reached.
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Prepare the new chunk by including overlapping sentences from the end of the current chunk.
            overlapping_chunk = []
            overlap_count = 0
            # Reverse iterate over current_chunk to accumulate overlap.
            for sent in reversed(current_chunk):
                sent_word_count = len(sent.split())
                overlap_count += sent_word_count
                overlapping_chunk.insert(0, sent)
                if overlap_count >= overlap:
                    break
            # Start the new chunk with overlapping sentences and the current sentence.
            current_chunk = overlapping_chunk.copy()
            current_chunk.append(sentence)
            current_word_count = sum(len(s.split()) for s in current_chunk)
    
    # Append the last chunk if any exists.
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# -------------------------------
# 2. Embedding-based Retrieval
# -------------------------------

class EmbeddingRetriever:
    """
    An embedding-based retriever that encodes text chunks and retrieves them based on cosine similarity.
    """
    def __init__(self, model_name: str = "t5-small"):
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
            query (str): The user's query.
            top_k (int): Number of results to return.
        
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
    def __init__(self, model_name: str = "t5-base"):
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
    financial_keywords = [
        "revenue", "profit", "earnings", "financial", "statement", "balance sheet",
        "income", "cash flow", "quarter", "year", "growth", "sales", "dividend",
        "expenditure", "expense", "operating", "margin", "investment", "capital",
        "asset", "assets", "liability", "liabilities", "turnover", "cost", "forecast",
        "order", "volume", "gross", "net", "ebitda", "cogs", "sg&a", "equity",
        "depreciation", "amortization", "krw", "billion", "thousand", "wholesale",
        "retail", "eco-friendly", "evs", "hevs", "phevs", "fcevs"
    ]
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in financial_keywords):
        return True, ""
    else:
        return False, "Query does not appear to be related to financial statements. Please ask questions about the company's financials."

# -------------------------------
# 6. Advanced RAG Implementation: Adaptive Retrieval
# -------------------------------

def combined_search(query: str, emb_retriever: EmbeddingRetriever, bm25_retriever: BM25Retriever, top_k: int = 5):
    """
    Combine the search results from embedding-based and BM25-based retrieval methods.
    
    This is the core of the "Advanced RAG" with Adaptive Retrieval:
      - It first retrieves the top `top_k` candidates using the embedding-based method (semantics via cosine similarity).
      - In parallel, the BM25 retriever obtains candidates based on exact keyword matches.
      - The BM25 scores are normalized (by dividing by the maximum BM25 score) and then added to the embedding-based scores.
      - The combined scores of candidate chunks are used to rank and select the most relevant pieces of the financial report.
      
    Parameters:
        query (str): The user query.
        emb_retriever (EmbeddingRetriever): Instance of embedding-based retriever.
        bm25_retriever (BM25Retriever): Instance of BM25 keyword-based retriever.
        top_k (int): Number of top results to consider from each method (default: 5).

    Returns:
        list: Sorted list of tuples (chunk, combined_score) in descending order of relevance.
    """
    emb_results = emb_retriever.search(query, top_k=top_k)
    bm25_results = bm25_retriever.search(query, top_k=top_k)
    
    scores = {}
    # Incorporate embedding-based scores.
    for chunk, score in emb_results:
        scores[chunk] = scores.get(chunk, 0) + score
        
    # Normalize BM25 scores and incorporate them.
    if bm25_results:
        max_bm25 = max(score for _, score in bm25_results)
        for chunk, score in bm25_results:
            normalized_score = score / max_bm25 if max_bm25 > 0 else 0
            scores[chunk] = scores.get(chunk, 0) + normalized_score
            
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_chunks

# -------------------------------
# 7. Chatbot UI with Streamlit and Confidence Scoring
# -------------------------------

def logistic_confidence(score, midpoint=0.45, steepness=10):
    """
    Convert a raw cosine similarity score to a confidence score using a logistic function.
    
    Parameters:
        score (float): The raw cosine similarity score.
        midpoint (float): The value of the score at which the logistic function returns 0.5.
        steepness (float): Controls how steep the transition is from low to high confidence.
        
    Returns:
        float: A confidence score between 0 and 1.
    """
    return 1 / (1 + math.exp(-steepness * (score - midpoint)))

def main():
    """
    The main function to run the RAG Chatbot using Streamlit.
    
    It performs the following steps:
      1. Validates the user query using a guardrail.
      2. Loads and preprocesses the financial report.
      3. Initializes the retrievers and response generator.
      4. Retrieves relevant context via Adaptive Retrieval by combining embedding-based and BM25 results.
      5. Generates a response and computes a confidence score.
      6. Displays a loader while processing the query.
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
        with st.spinner("Processing query..."):
            # Validate query using guardrail
            is_valid, message = validate_query(user_input)
            if not is_valid:
                response = "Guardrail Alert: " + message
                confidence = 0.0
            else:
                # Load and preprocess the financial report
                data_path = os.path.join("data", "financial_report.txt")
                report_text = load_financial_report(data_path)
                # Process text into chunks using the chunk merging strategy.
                chunks = chunk_text(report_text, max_length=150, overlap=75)
                
                # Initialize retrieval methods and response generation.
                emb_retriever = EmbeddingRetriever()
                emb_retriever.add_documents(chunks)
                bm25_retriever = BM25Retriever(chunks)
                response_generator = ResponseGenerator()
                
                # Retrieve context using Advanced RAG (Adaptive Retrieval).
                results = combined_search(user_input, emb_retriever, bm25_retriever, top_k=5)
                top_chunks = [chunk for chunk, score in results[:3]]
                context = "\n".join(top_chunks)
                response = response_generator.generate_response(user_input, context)
                
                # Compute confidence score using logistic scaling on the top 10 cosine similarity scores.
                emb_results = emb_retriever.search(user_input, top_k=10)
                raw_scores = [score for _, score in emb_results]
                logistic_scores = [logistic_confidence(s) for s in raw_scores]
                sorted_logistic = sorted(logistic_scores, reverse=True)
                if len(sorted_logistic) >= 3:
                    confidence = sum(sorted_logistic[:3]) / 3.0
                elif sorted_logistic:
                    confidence = sorted_logistic[0]
                else:
                    confidence = 0.0
            
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": response,
                "confidence": confidence
            })
    
    for chat in st.session_state.chat_history:
        st.markdown(f"**User:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")
        st.markdown(f"**Confidence Score:** {chat['confidence']:.2f}")
    
if __name__ == "__main__":
    main() 
