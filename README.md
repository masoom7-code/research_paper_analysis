# Semantic Research Paper Search Engine

An NLP-powered search system that retrieves research papers based on meaning rather than keywords. Built using Sentence Transformers for embedding generation and FAISS for fast vector similarity search.

## Overview

Traditional keyword search fails when the query and documents use different vocabulary for the same concept. This project addresses that limitation by converting paper abstracts into dense vector embeddings that capture semantic meaning and using FAISS to retrieve the most relevant papers for a natural language query.

The system also includes an AI-powered summarization layer that generates concise summaries of retrieved research papers using a pre-trained BART model.

Built as part of the CBSOT Summer Internship 2026.

## How It Works

1. Data Loading & Cleaning - Load the ML-ArXiv-Papers dataset, merge titles with abstracts, and remove formatting artifacts.
2. Embedding Generation - Use `all-MiniLM-L6-v2` to convert each paper into a 384-dimensional vector.
3. Index Building - Store embeddings in a FAISS index with L2 normalization for cosine similarity search.
4. Query Processing - Convert user queries into vectors using the same Sentence Transformer model.
5. Retrieval - Find the top-K most semantically similar papers using FAISS inner product search.
6. Summarization - Generate concise AI summaries of retrieved abstracts using DistilBART.

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

Clone the repository:

```bash
git clone https://github.com/masoom7-code/semantic-research-paper-search.git
```

Navigate to the project directory:

```bash
cd semantic-research-paper-search
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run the Project

The project uses Jupyter Notebook to execute the data processing, embedding generation, semantic search, and summarization pipeline.

### Step 1: Install Jupyter Notebook

If Jupyter Notebook is not installed, run:

```bash
pip install notebook
```

### Step 2: Start Jupyter Notebook

From the project directory, run:

```bash
jupyter notebook
```

A Jupyter Notebook interface should open automatically in your web browser.

If it does not open automatically, copy the localhost URL displayed in the terminal and open it in your browser.

### Step 3: Run the EDA Notebook

Open:

```text
EDA.ipynb
```

Run all notebook cells in order.

This notebook:

* Loads the ML-ArXiv-Papers dataset
* Performs data exploration and cleaning
* Combines paper titles and abstracts
* Generates 384-dimensional semantic embeddings using `all-MiniLM-L6-v2`
* Saves the generated embeddings and cleaned dataset inside the `data/` directory

Generated files:

```text
data/
├── arxiv_embeddings.npy
└── cleaned_arxiv_papers.csv
```

Embedding generation may take some time depending on the system hardware.

### Step 4: Run the Semantic Search Notebook

After completing `EDA.ipynb`, open:

```text
Search_Engine.ipynb
```

Run all notebook cells in order.

This notebook:

* Loads the generated embeddings
* Builds the FAISS vector index
* Processes natural language search queries
* Retrieves semantically similar research papers
* Generates concise AI summaries using DistilBART

### Step 5: Search for Research Papers

Use the `search_and_summarize()` function to search for research papers.

Example:

```python
search_and_summarize("deep learning in medical science", k=5)
```

You can replace the query with any research topic.

Example queries:

```python
search_and_summarize("artificial intelligence in healthcare", k=5)

search_and_summarize("transformer models for natural language processing", k=5)

search_and_summarize("computer vision for autonomous vehicles", k=5)
```

The system returns:

* Research paper titles
* Paper abstracts
* Semantic similarity scores
* AI-generated summaries

## Tech Stack

* Embeddings: Sentence Transformers (`all-MiniLM-L6-v2`)
* Vector Search: FAISS (Facebook AI Similarity Search)
* Summarization: Hugging Face Transformers (DistilBART)
* Keyword Extraction: KeyBERT
* Data Processing: Pandas, NumPy
* ML Utilities: Scikit-learn

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

Masoom Perwez Alam

Developed as part of the CBSOT Summer Internship 2026.
