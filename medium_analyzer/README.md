# Overview
This project explores two distinct methods of implementing a document analyzer using Large Language Models (LLM). It's an experimental venture into creating a Question-Answering (QA) agent that leverages the capabilities of LLMs.

# Implementations
1. Pinecone-based Analyzer (pinecore.py):
* Utilizes OpenAI embeddings.
* Leverages the vector database of Pinecone for document retrieval.
2. Local Storage-based Analyzer (local.py):
* Manages and retrieves documents using Faiss.

# Getting Started
## Prerequisites
1. Ensure you have Python 3.11.0 or higher installed.
2. If you're working with the Pinecone implementation, make sure to have Pinecone SDK set up and a valid API key for OpenAI.
3. Fill the API keys and environment name in `constants.py`
4. Familiarize yourself with the structure of the files and the underlying architecture of LLMs.

## Installation
1. Clone the repository:
```
git clone [repository-url]
```

2. Navigate to the project directory:
```
cd medium_analyzer
```

3. Install the required packages:
```python
pip install -r requirements.txt
```

# Usage
* __For Pinecone-based Analyzer:__
```
python pinecore.py
```
* __For Local Storage-based Analyzer:__
```
python faiss.py
```