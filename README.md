#  RAG-QA Model

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for Question Answering tasks. The model leverages the power of dense retrieval and language generation to answer questions based on a provided corpus of documents.

##  Features

- End-to-end RAG pipeline
- Dense passage retrieval using FAISS
- Context chunking with semantic search
- Answer generation using pre-trained transformer models
- Modular and easy to customize

---

## Directory Structure

```
Rag-QAModel/
├── datset.pdf             # the document 
├── qa_system.py           # The main file 
└── requirements.txt       # Requirements necessary 
```

---

## Installation

Make sure you have **Python 3.8+** installed.

```bash
git clone https://github.com/grimreapermanasvi/Rag-QAModel.git
cd Rag-QAModel
pip install -r requirements.txt
```

---

## Usage

Run the RAG pipeline:

```bash
python qa_system.py
```

Sample output:

```
> Retrieved Contexts: [...top documents...]
> Generated Answer: Retrieval-augmented generation (RAG) is...
```

---

## Example Use Case

You can load your own dataset (PDFs, text files, scraped content) and plug it into the retriever. The pipeline is adaptable to various domains like:

- Medical QA
- Legal document analysis
- Academic research tools

---

## Tech Stack

- Hugging Face Transformers
- FAISS (Facebook AI Similarity Search)
- Python
- Pandas, Numpy
- tqdm, argparse

---

## Author

**Manasvi Srivastava**  
[GitHub](https://github.com/grimreapermanasvi)  
[LinkedIn](https://www.linkedin.com/in/manasvi-sri/) 

---

## Star This Repo

If you found this useful, consider starring ⭐ the repo to show your support!

---
