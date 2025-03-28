# Article-Summarizer-Tool

Host Link :- https://huggingface.co/spaces/tanishq04/Article-Summarizer-Tool   ( May not work after all free token trail get used )

## Overview
The **Article-Summarizer-Tool** is a Streamlit-based web application that allows users to extract and summarize information from multiple online articles. Users can input article URLs, process the content, generate embeddings using FAISS, and then query the tool to retrieve relevant information with answers generated by an LLM (Mistral-7B-Instruct-v0.3 from Hugging Face).

## Features
- Accepts up to three URLs from users.
- Loads and processes text from given URLs.
- Splits extracted text into manageable chunks for better retrieval.
- Embeds processed text using **sentence-transformers/all-MiniLM-L6-v2**.
- Stores embeddings in a FAISS vector store.
- Allows users to ask questions about the processed content.
- Uses **Mistral-7B-Instruct-v0.3** from Hugging Face for generating responses based on retrieved content.

## Technologies Used
- **Python** (Core scripting language)
- **Streamlit** (Web app framework)
- **Hugging Face Transformers** (LLM & Embeddings)
- **FAISS** (Vector search)
- **LangChain** (RetrievalQA processing)
- **UnstructuredURLLoader** (Extracts content from web pages)
- **dotenv** (Manages API keys)
- **Pickle** (Serialization of FAISS index)

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/Article-Summarizer-Tool.git
   cd Article-Summarizer-Tool
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up the Hugging Face API Key:
   - Create a `.env` file in the project directory.
   - Add the following line:
     ```sh
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
     ```
4. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit web app.
2. Enter up to **three article URLs** in the sidebar.
3. Click **"Process URLs"** to load and store the articles.
4. Once processed, enter a **question** in the text input field.
5. The tool retrieves relevant content, sends it to the LLM, and displays the summarized answer along with sources.

## Troubleshooting
### Issue: API Key Not Found
- Ensure you have added your **Hugging Face API Key** in the `.env` file.
- Restart the Streamlit application after making changes.

### Issue: No Chunks Found After Processing URLs
- Some URLs may not be supported by `UnstructuredURLLoader`.
- Try different sources.

### Issue: Unexpected Response Format from LLM
- The response from `Hugging Face Hub` might have changed. Debug by printing `response` before processing it.

