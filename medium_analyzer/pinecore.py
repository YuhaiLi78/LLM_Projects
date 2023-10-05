from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone

from constants import *

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if __name__ == "__main__":
    loader = TextLoader("./data/mediumblog1.txt")
    document = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=200)
    
    texts = text_splitter.split_documents(documents=document)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Pinecone.from_documents(texts, embeddings, index_name=PINECONE_INDEX)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), 
        chain_type='stuff', 
        retriever=docsearch.as_retriever(), 
        return_source_documents=True
    )

    query = "What is a vector DB? Give me a 15-word answer for beginer"
    result = qa({'query': query})
    print(result)