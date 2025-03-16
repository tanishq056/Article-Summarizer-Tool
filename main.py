import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_key:
    st.error("Hugging Face API Key not found! Please add it to your .env file.")
    st.stop()

# Initialize Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 0.7, "max_length": 500}
)

st.title("Article-Summarizer-Tool")
st.sidebar.title("🔗Article URLs")

# Collect URLs
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("🔄 Process URLs")
file_path = "faiss_store_hf.pkl"

if process_url_clicked:
    if not any(urls):
        st.sidebar.warning("⚠️ Please enter at least one valid URL.")
    else:
        with st.status("Processing URLs...", expanded=True) as status:
            st.write("✅ **Loading data from URLs...**")
            loader = UnstructuredURLLoader(urls=[url for url in urls if url])
            data = loader.load()

            st.write("✅ **Splitting text into chunks...**")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,    # Reduced to avoid long document issues
                chunk_overlap=150  # Helps with continuity
            )

            docs = []
            for doc in data:
                try:
                    chunks = text_splitter.split_documents([doc])
                    docs.extend(chunks)
                except ValueError as e:
                    st.warning(f"⚠️ Skipping document: {str(e)}")

            if not docs:
                st.error("❌ No valid document chunks found. Please try different URLs.")
                st.stop()

            st.write(f"✅ **Generated {len(docs)} chunks successfully!**")

            st.write("✅ **Generating embeddings...**")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore_hf = FAISS.from_documents(docs, embeddings)

            # Save FAISS index
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_hf, f)

            status.update(label="✅ Processing Complete!", state="complete")

# Process user query
query = st.text_input("🔎 Ask a Question:")
# if query:
#     if not os.path.exists(file_path):
#         st.warning("⚠️ Please process URLs first!")
#     else:
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)
#             chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#             result = chain({"question": query}, return_only_outputs=True)

#             st.header("📝 Answer")
#             st.write(result["answer"])

#             # Display sources
#             sources = result.get("sources", "").strip()
#             if sources:
#                 st.subheader("📌 Sources:")
#                 for source in sources.split("\n"):
#                     st.write(f"🔗 {source}")
# if query:
#     if not os.path.exists(file_path):
#         st.warning("⚠️ Please process URLs first!")
#     else:
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)
#             retrieved_docs = vectorstore.similarity_search(query, k=5)

#             # Combine retrieved docs into a single context
#             context = "\n\n".join([doc.page_content for doc in retrieved_docs])

#             #st.write(f"🔍 **Context Retrieved:**\n{context}")

#             # Pass context to LLM
#             response = llm.invoke(f"Answer the following based on the given context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:")
#             response_text = response['generations'][0][0]['text']
#             qa_part = "\n".join([line for line in response_text.split("\n") if "Question:" in line or "Answer:" in line])


#             st.header("📝 Answer")
#             st.write(qa_part)
if query:
    if not os.path.exists(file_path):
        st.warning("⚠️ Please process URLs first!")
    else:
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retrieved_docs = vectorstore.similarity_search(query, k=5)

            # Combine retrieved docs into a single context
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # Debug: Print context if needed
            # st.write(f"🔍 **Context Retrieved:**\n{context}")

            # Pass context to LLM
            response = llm.invoke(f"Answer the following based on the given context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:")

            # Check response type before processing
            if isinstance(response, dict) and 'generations' in response:
                response_text = response['generations'][0][0]['text']
            elif isinstance(response, str):
                response_text = response  # Directly use if it's already a string
            else:
                st.error("⚠️ Unexpected response format from LLM!")
                response_text = "No valid response received."

            # Extract only Q&A part
            qa_part = "\n".join([line.replace("Question:", "\n Question:").replace("Answer:", "\n Answer: ") for line in response_text.split("\n") if "Question:" in line or "Answer:" in line])

            st.header("📝 Answer")
            st.write(qa_part)


            




