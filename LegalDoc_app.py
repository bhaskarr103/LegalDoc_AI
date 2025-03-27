import os
import streamlit as st
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def load_vectorstore():
    """Load the FAISS vector store only once."""
    embedding_model = HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L3-v2')  # Faster model
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def get_vectorstore():
    """Retrieve the loaded FAISS vector store from session state."""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorstore()
    return st.session_state.vectorstore

def set_custom_prompt():
    return PromptTemplate(
        template="""
        Use the given legal document and database to answer user's legal questions.
        If the uploaded document contains relevant legal points, prioritize those.
        If no relevant law is found in the document, check the pre-stored database.
        If you still don't know, say: "No relevant legal provision found."
        Don't fabricate legal information.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk.
        """,
        input_variables=["context", "question"]
    )

def load_llm():
    HF_TOKEN = os.getenv("HF_TOKEN")
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )

def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

@st.cache_resource
def create_embeddings_from_text(_text_chunks):
    """Generate embeddings but cache results to avoid redundant processing."""
    embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")  # Faster model
    
    # Check if embeddings already exist in session state
    if "cached_embeddings" not in st.session_state:
        st.session_state.cached_embeddings = {}
    
    # Convert Documents to a hashable format (tuple of text content)
    text_hashable = tuple(doc.page_content for doc in _text_chunks)
    doc_hash = hash(text_hashable)
    
    if doc_hash in st.session_state.cached_embeddings:
        return st.session_state.cached_embeddings[doc_hash]
    
    # Start timing
    start_time = time.time()
    
    vectorstore = FAISS.from_documents(_text_chunks[:10], embedding_model)  # Process only 10 pages for now
    
    # End timing
    elapsed_time = time.time() - start_time
    st.write(f"✅ Embedding generation took {elapsed_time:.2f} seconds")
    
    # Cache result
    st.session_state.cached_embeddings[doc_hash] = vectorstore
    return vectorstore

def process_uploaded_pdf(uploaded_file):
    if uploaded_file:
        save_path = os.path.join("data", uploaded_file.name)  # Store in 'data/' folder
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Extract text from PDF
        extracted_data = extract_text_from_pdf(save_path)

        # Chunk the extracted text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.split_documents(extracted_data)

        # Generate embeddings
        return create_embeddings_from_text(text_chunks)
    return None

# ------------- Streamlit UI ---------------
def main():
    st.title("Legal Advisor Bot!")

    uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])

    # Process uploaded PDF
    user_vectorstore = None
    if uploaded_file:
        st.success("Processing document...")
        user_vectorstore = process_uploaded_pdf(uploaded_file)

    # Retrieve stored legal knowledge
    vectorstore = get_vectorstore()

    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Enter your legal question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            retrievers = [vectorstore.as_retriever(search_kwargs={'k': 3})]
            
            # If user uploaded a document, add its retriever
            if user_vectorstore:
                retrievers.append(user_vectorstore.as_retriever(search_kwargs={'k': 3}))

            # Combine retrievers
            combined_retriever = retrievers[0]
            if len(retrievers) > 1:
                from langchain.retrievers import EnsembleRetriever
                combined_retriever = EnsembleRetriever(retrievers=retrievers)

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=combined_retriever,
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            formatted_response = f"### LegalDoc_AI Response\n\n{result}\n\n---\n\n### 📚 Source Documents:\n"
            for i, doc in enumerate(source_documents, start=1):
                formatted_response += f"**{i}. Source:** `{doc.metadata.get('source', 'Unknown')}`, **Page:** {doc.metadata.get('page', 'N/A')}\n\n"
                formatted_response += f"{doc.page_content.strip()}\n\n"

            st.chat_message('assistant').markdown(formatted_response)
            st.session_state.messages.append({'role': 'assistant', 'content': formatted_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()