import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS


from langchain.embeddings import HuggingFaceEmbeddings


from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Query US Public Laws (RAG)")

faiss_index_path = "faiss_index"
if not os.path.exists(faiss_index_path):
    st.error(f"Vector database not found at '{faiss_index_path}'. Run the vector storage program first.")
    st.stop()

if 'vectors' not in st.session_state:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.vectors = FAISS.load_local(faiss_index_path, embeddings=embeddings)

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")


prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant specialized in US Public Laws. 
    Answer the user's question strictly based on the provided context.

    If the context does not contain relevant information, respond with: 
    "I couldn't find relevant information in the available documents."

    **Context:**
    {context}

    **User Question:** {input}

    **Instructions:**
    - Keep your response factual and concise.
    - Do not make up information beyond the provided context.
    - If necessary, cite relevant parts of the documents retrieved.

    **Response:**"""
)

retriever = st.session_state.vectors.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

query = st.text_input("Enter your legal query:")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching for relevant laws..."):
            start_time = time.process_time()
            
            response = retrieval_chain.invoke({'input': query})

            st.write("### Response:")
            st.write(response['answer'])
            st.write(f"Processing time: {time.process_time() - start_time:.2f} seconds")

            with st.expander("Retrieved Documents (Context)"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Document {i + 1}:**")
                    st.write(doc.page_content)
                    st.divider()
