# RAG Chatbot for Financial Statements

This repository contains an end-to-end Retrieval-Augmented Generation (RAG) Chatbot that answers financial queries based on a company's financial statements (last two years). The chatbot leverages both embedding-based techniques and BM25 keyword retrieval (adaptive retrieval) to extract the most relevant context from a financial report and then generates responses using a language model.

## Features

- **Data Collection & Preprocessing:**  
  Loads a financial report and splits it into overlapping text chunks using a "Chunk Merging" strategy to preserve contextual information.

- **Embedding-based Retrieval:**  
  Encodes text chunks using SentenceTransformers (using the `all-MiniLM-L6-v2` model) and retrieves relevant segments via cosine similarity.

- **BM25 Retrieval (Keyword-based):**  
  Uses BM25 (via the `rank-bm25` library) to perform keyword-based search over the chunks, which is then combined with the embedding-based search for adaptive retrieval.

- **Adaptive Retrieval:**  
  Combines scores from both the BM25 and embedding retrieval methods to improve the relevance of the retrieved context.

- **Guardrail Implementation:**  
  Validates that user queries are financially related by checking for common financial keywords.

- **Chatbot UI using Streamlit:**  
  Provides a web interface that supports multi-turn conversation. The interface shows a loader (spinner) when processing each query, and displays every user query, the generated response, and an associated confidence score (measured as the average cosine similarity from the embedding retrieval).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.6 or newer installed, then install the required packages using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:

   - streamlit
   - sentence-transformers
   - rank-bm25
   - transformers
   - numpy

## Data Preparation

- Create a directory called `data` in the project root.
- Place your financial report in a file named `financial_report.txt` inside the `data` folder.

## Running the App

Launch the chatbot application using Streamlit:

```bash
streamlit run rag_chatbot.py
```

This command will open your default web browser with the chatbot interface. When you enter a query, a loader spinner will be displayed while the app processes your request.

## Deployment to Streamlit Community Cloud

To deploy your app to the Streamlit Community Cloud:

1. **Push your code to GitHub:**  
   Ensure your repository includes the following:

   - `rag_chatbot.py` (the main app file)
   - `requirements.txt` (list of dependencies)
   - `data/financial_report.txt` (or instructions to create it)

2. **Deploy on Streamlit Community Cloud:**
   - Sign in to [Streamlit Community Cloud](https://share.streamlit.io) with your GitHub account.
   - Create a new app by selecting your repository, setting the branch (e.g., `main`), and specifying the file path to `rag_chatbot.py`.
   - Click **Deploy** to launch your app. You will receive a unique URL for your deployed chatbot.

## Project Structure

```
.
├── data
│   └── financial_report.txt    # Your financial report file
├── rag_chatbot.py              # Main application code
├── requirements.txt            # List of dependencies
└── README.md                   # This file
```

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/)
- [Rank-BM25](https://pypi.org/project/rank-bm25/)
- [Transformers by Hugging Face](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)

## License

This project is licensed under the MIT License.
