import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Set the page configuration at the start
st.set_page_config(page_title="Chat with PDF using Gemini")

# Add CSS for the background image
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://cdn.pixabay.com/photo/2016/11/29/06/15/plans-1867745_960_720.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}

[data-testid="stSidebar"] {
background-image: url("https://cdn.pixabay.com/photo/2015/01/21/14/14/imac-606765_960_720.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables")
else:
    genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say, "answer is not available in the context".
    Context: {context}
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Allow dangerous deserialization as we trust the source of the data
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return
    
    try:
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        return
    
    chain = get_conversational_chain()
    
    try:
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        # Highlight the reply
        st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'><strong>Reply:</strong> {response['output_text']}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error during chain execution: {e}")

def main():
    st.header("Chat with PDF using Gemini")
    user_question = st.text_input("Ask a Question from the PDF files")
    
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    st.error("No text extracted from PDF files.")
                    return
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete.")

if __name__ == "__main__":
    main()
