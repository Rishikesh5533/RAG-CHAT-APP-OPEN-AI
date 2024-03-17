from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader,PyPDFDirectoryLoader,Docx2txtLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings

print("done")

import os
os.environ["OPENAI_API_KEY"] = "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"
persist_directory = "./pages/db"

def main():
    pdf_loader = DirectoryLoader('.pages/Reports/Pdf/', glob="**/*.pdf")
    txt_loader = DirectoryLoader('.pages/Reports/txt/', glob="**/*.txt")
    word_loader = DirectoryLoader('.pages/Reports/doc/', glob="**/*.docx")

    loaders = [pdf_loader, txt_loader, word_loader]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    print(f"Total number of documents: {len(documents)}")
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    documents = text_splitter.split_documents(documents)
    db = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    db.persist()
    db = None

if __name__ == "__main__":
    main()
 
