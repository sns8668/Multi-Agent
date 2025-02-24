import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import tempfile

# Initialize OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Streamlit app title
st.title("📚 Multi-Agent Research Assistant")

# File uploader
uploaded_file = st.file_uploader("Upload a Research Paper (PDF format)", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load and process the document
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Combine the text from all pages
    file_content = " ".join([doc.page_content for doc in documents])

    # Split the content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(file_content)

    # Convert chunks into Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Summarization agent
    st.header("1️⃣ Summarization Agent")
    with st.spinner("Summarizing the document..."):
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
        summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = summarize_chain.run(docs)
        st.success("Summarization complete!")
        st.text_area("Summary", summary, height=200)

    # Critique agent
    st.header("2️⃣ Critique Agent")
    with st.spinner("Performing critique analysis..."):
        # Example critique logic (replace with API-based retrieval and critique logic)
        critique_prompt = PromptTemplate(
            input_variables=["summary"],
            template="""
            Based on the following summary, provide a critique. Highlight strengths, weaknesses, and suggest areas for improvement:
            {summary}
            """
        )
        critique_chain = LLMChain(llm=llm, prompt=critique_prompt)
        critique = critique_chain.run(summary)
        st.success("Critique complete!")
        st.text_area("Critique", critique, height=200)

    # Refinement agent
    st.header("3️⃣ Refinement Agent")
    with st.spinner("Refining the summary..."):
        refinement_prompt = PromptTemplate(
            input_variables=["summary", "critique"],
            template="""
            Refine the following summary using the provided critique. Ensure clarity, logical flow, and concise language:
            Summary:
            {summary}

            Critique:
            {critique}

            Refined Summary:
            """
        )
        refinement_chain = LLMChain(llm=llm, prompt=refinement_prompt)
        refined_summary = refinement_chain.run({"summary": summary, "critique": critique})
        st.success("Refinement complete!")
        st.text_area("Refined Summary", refined_summary, height=200)

    # Clean up temporary file
    os.unlink(temp_file_path)

else:
    st.info("Upload a PDF to get started!")


