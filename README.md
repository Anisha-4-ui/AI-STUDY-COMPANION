NADS: AI STUDY COMPANION - INTERACTIVE DOCUMENT ANALYSIS & TUTOR

Version 1

NADS (AI Study Companion) is an intelligent, multimodal learning assistant that helps users study, explore topics, and test their understanding through quizzes, voice interaction, and document-based learning. It combines large language models, retrieval-augmented generation (RAG), web search, and speech features into a single interactive application.

The app is built with Streamlit for the frontend and Python for the backend, with flexible support for both cloud-based and local LLMs.

## Key Capabilities

Chat-based learning with AI
PDF-based question answering using RAG
Automatic quiz generation and evaluation
Web search and scraping for live educational content
Voice input (speech-to-text) and voice output (text-to-speech)
Downloadable AI-generated content as PDFs
Model evaluation using standard NLP metrics

## Tech Stack

Frontend:
Streamlit – Interactive web application UI
Custom CSS – Enhanced styling and layout control

Backend / Core Logic:
Python 3.x – Core language
python-dotenv – Environment variable and API key management

AI & NLP:
Google Gemini (google-generativeai, langchain-google-genai) – Primary LLM for chat, summarization, and quiz generation
LangChain – Prompt management, LLM chaining, and RAG orchestration
Ollama (langchain-ollama) – Local embedding and LLM support
Sentence Transformers – Text embeddings for semantic search

Retrieval-Augmented Generation (RAG):
FAISS – Fast vector similarity search
pypdf – PDF text extraction

Quiz System:
Custom Python modules – Quiz generation, scoring, and feedback

Web Scraping & Search:
SerpAPI – Google search restricted to educational sources
BeautifulSoup – HTML parsing and content extraction

Voice Features:
SpeechRecognition – Speech-to-text
pyttsx3 – Text-to-speech
pyaudio – Microphone and audio processing

PDF Generation:
ReportLab – Create downloadable, formatted PDFs

Parsers:
mistune, html2text – Markdown, HTML, and text conversion

Evaluation & Testing:
rouge-score – Text generation evaluation
scikit-learn – Similarity metrics and evaluation utilities

## Features

Dual Knowledge Modes
Document Mode (With PDFs)
Extracts text from uploaded PDFs
Generates embeddings and stores them in FAISS
Answers questions using retrieval-augmented generation
Live Mode (Without PDFs)
Performs real-time web search and scraping
Filters and summarizes educational content
Provides concise, relevant answers with references
Intelligent Context Switching
The system automatically decides whether to use document-based retrieval or live web data based on user input and available resources.
Quiz Generation and Evaluation
Automatically generates quizzes from PDFs or topics
Evaluates answers and provides feedback
Supports scoring and performance analysis
Voice-Enabled Interaction
Accepts spoken queries through a microphone
Responds with natural-sounding voice output
Improves accessibility and engagement
Model Evaluation
Uses ROUGE and cosine similarity metrics
Helps assess response quality and relevance

## Project Structure

AI-Tutor/
├── aiFeatures/
│ ├── **pycache**/
│ ├── quiz_system.cpython-313.pyc
│ └── python/
│ ├── ai_assistant.py               # Core decision-making and query routing
│ ├── ai_response.py                # LLM response handling
│ ├── evaluation.py                 # Model evaluation logic
│ ├── evaluation_dataset.py         # Evaluation datasets
│ ├── quiz_system.py                # Quiz generation and scoring
│ ├── rag_pipeline.py               # RAG pipeline implementation
│ ├── speech_to_text.py             # Voice input handling
│ ├── text_to_speech.py             # Voice output handling
│ ├── web_scraper_tool.py           # Search and scraping utilities
│ └── web_scraping.py               # Web content extraction
│
├── data/                           # Stored data and intermediate outputs
├── env/                            # Virtual environment (optional)
├── .env                            # API keys and environment variables
├── .gitignore                      # Git ignore rules
├── app.py                          # Streamlit application entry point
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies

## Use Cases

Students studying from textbooks or PDFs
Self-learners exploring new topics
Educators generating quizzes and summaries
Voice-based hands-free learning

## Future Enhancements

User authentication and progress tracking
Multi-language support
Persistent vector storage
Advanced analytics dashboard

## License

This project is intended for educational and research purposes.

## Getting Started

cd AI-Tutor

## Set Up Virtual Environment

python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

## Install Dependencies

pip install -r requirements.txt

## Configure Environment Variables

Create a `.env` file and add:

GEMINI_API_KEY="your_gemini_key"
SERP_API_KEY="your_serpapi_key"
OLLAMA_MODEL=mxbai-embed-large

## Run the Application

streamlit run app.py
Streamlit will display the local URL in the terminal (usually http://localhost:8501).

## Ollama Setup

To use **Ollama embeddings** for document chunking and vector representation, follow these steps:

### 1. Install Ollama

Download and install Ollama from the official site:

👉 [https://ollama.com/download](https://ollama.com/download)

After installation, ensure it is accessible in your terminal:

ollama --version

### 2. Pull the Required Model

Pull a supported embedding model like `mxbai-embed-large` or any other compatible model:

ollama pull mxbai-embed-large

### 3. Start Ollama Server (if required)

Ollama typically runs as a background service. If not, you can start it manually:

ollama serve

### 4. Integration in AI Tutor

The application uses Ollama to embed chunks of PDF/text using:

- `mxbai-embed-large` or any embedding model you configure
- Embeddings are stored in FAISS vector store and used for cosine similarity-based retrieval

Ensure your `.env` or config file includes proper references to use Ollama embeddings.

OLLAMA_MODEL=mxbai-embed-large

You’re now ready to use Ollama with AI Tutor🌟

