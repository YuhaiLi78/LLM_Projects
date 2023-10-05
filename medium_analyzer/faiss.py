from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from constants import OPENAI_API_KEY

if __name__ == "__main__":
    file_path = './data/react.pdf'
    index_name = 'faiss_index_react'
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=30,
                                          separator='\n')
    
    docs = text_splitter.split_documents(documents=documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    # vectorstore.save_local(index_name)

    new_vectorstore = FAISS.load_local(index_name, embeddings)
    qa = RetrievalQA(llm=OpenAI(),
                     chain_type="stuff",
                     retriever=new_vectorstore.as_retriever())
    
    res = qa.run("Give me the gist of ReAct in 3 sentences")
    print(res)

