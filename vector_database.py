import os
import pickle
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

st.title("Vector Database Creation")

pdf_dir = "./US_PublicLaws"
vector_dir = "./vectorstore"

if not os.path.exists(pdf_dir):
    st.error(f"Directory '{pdf_dir}' not found. Please check the folder path.")
    st.stop()

if not os.path.exists(vector_dir):
    os.makedirs(vector_dir)
    st.info(f"Created vectorstore directory at {vector_dir}")

def create_vector_database():
    try:
        # embeddings = OpenAIEmbeddings(
        #     model="text-embedding-3-small", #alternative text-embedding-3-large	could cost a little bit more
        #     show_progress_bar=True
        # )

        #Let's see with huggingface model

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"        )

        st.write(f"Loading PDFs from directory: {pdf_dir}")
        loader = PyPDFDirectoryLoader(pdf_dir)
        documents = loader.load()

        if not documents:
            st.error(f"No PDF documents found in {pdf_dir}")
            st.stop()

        st.write(f"Found {len(documents)} documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        
        final_file = text_splitter.split_documents(documents)
        
        st.write(f"Created {len(final_file)} chunks from the documents")
        
        vectors = FAISS.from_documents(final_file, embeddings)

        faiss_index_path = os.path.join(vector_dir, "faiss_index")
        vectors.save_local(faiss_index_path)



        pkl_path = os.path.join(vector_dir, "public_law.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(final_file, f)

        st.success(f"""
        Vector database created successfully!
        - FAISS index saved to: {faiss_index_path}
        - Document chunks saved to: {pkl_path}
        - Total chunks processed: {len(final_file)}
        """)

    except Exception as e:
        st.error(f"Error during vector embedding: {str(e)}")
        raise 

if st.button("Create and Store Vector Database"):
    with st.spinner("Processing PDFs and creating vector embeddings..."):
        create_vector_database()