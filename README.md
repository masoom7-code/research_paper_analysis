# Semantic Research Paper Search Engine

An NLP-powered search system that retrieves research papers based on meaning rather than keywords. Built using Sentence Transformers for embedding generation and FAISS for fast vector similarity search.

## Overview

Traditional keyword search fails when the query and documents use different vocabulary for the same concept. This project addresses that limitation by converting paper abstracts into dense vector embeddings that capture semantic meaning and using FAISS to retrieve the most relevant papers for a natural language query.

The system also includes an AI-powered summarization layer that generates concise summaries of retrieved research papers using a pre-trained BART model.

Built as part of the CBSOT Summer Internship 2026.

## How It Works

1. **Data Loading & Cleaning** - Load the ML-ArXiv-Papers dataset, merge titles with abstracts, and remove formatting artifacts.
2. **Embedding Generation** - Use `all-MiniLM-L6-v2` to convert each paper into a 384-dimensional vector.
3. **Index Building** - Store embeddings in a FAISS index with L2 normalization for cosine similarity search.
4. **Query Processing** - Convert user queries into vectors using the same Sentence Transformer model.
5. **Retrieval** - Find the top-K most semantically similar papers using FAISS inner product search.
6. **Summarization** - Generate concise AI summaries of retrieved abstracts using DistilBART.

## Project Structure

```
aiml_research/
├── EDA.ipynb             # Data exploration, cleaning, and embedding generation
├── Search_Engine.ipynb   # FAISS index setup, semantic search, and summarization
├── dataset.md            # Dataset reference
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup

```bash
git clone https://github.com/masoom7-code/semantic-research-paper-search.git
cd semantic-research-paper-search
pip install -r requirements.txt
```

## Usage

**Step 1:** Run `EDA.ipynb` to load the dataset, clean the data, and generate embeddings. This saves `arxiv_embeddings.npy` and `cleaned_arxiv_papers.csv` to the `data/` folder.

**Step 2:** Run `Search_Engine.ipynb` to build the FAISS index and search for research papers.

Example query:

```python
search_and_summarize("deep learning in medical science", k=5)
```

The system returns research paper titles, abstracts, similarity scores, and AI-generated summaries.

## Tech Stack

* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Search:** FAISS (Facebook AI Similarity Search)
* **Summarization:** Hugging Face Transformers (DistilBART)
* **Keyword Extraction:** KeyBERT
* **Data Processing:** Pandas, NumPy
* **ML Utilities:** Scikit-learn (cosine similarity)

## Dataset

The project uses the ML-ArXiv-Papers dataset from Hugging Face, containing thousands of machine learning research paper titles and abstracts.

Dataset: https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers

## Future Work

* Streamlit web interface for interactive semantic search
* PDF upload and research paper search capability
* Hybrid keyword and semantic search
* REST API for external integration
* Docker support for deployment

## Author

**Masoom Perwez Alam**

Developed as part of the **CBSOT Summer Internship 2026**.
