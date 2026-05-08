"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
TASK:
  1. Load your dataset, split into chunks, index with FAISS
  2. Build a RAG chain: retriever → prompt → LLM → output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all 50 questions → generates ≥ 50 LangSmith traces

DELIVERABLE: Open https://smith.langchain.com and confirm traces appear.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Environment setup ────────────────────────────────────────────────────
load_dotenv()

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "lab22_llmops")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# ── 2. LangChain + LangSmith imports ────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# ── 3. LLM and Embeddings Factory ───────────────────────────────────────────
def get_llm():
    # 1. Try Groq (Highly recommended for free tier)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print("  🤖 Using Groq (Llama 3.1)")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            groq_api_key=groq_key
        )

    # 2. Try Gemini
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        print("  🤖 Using Google Gemini (Free Tier)")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            google_api_key=google_key
        )
    
    # 4. Fallback to OpenAI
    print("  🤖 Using OpenAI (Fallback)")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )

def get_embeddings():
    # Default to local HuggingFace to avoid quota issues
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    if use_local:
        print("  🧠 Using Local HuggingFace Embeddings (Free)")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )

llm = get_llm()
embeddings = get_embeddings()

# ── 4. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.
    """
    data_path = Path("data/knowledge_base.txt")
    if not data_path.exists():
        data_path = Path("../data/knowledge_base.txt")
        if not data_path.exists():
            print(f"Error: knowledge_base.txt not found.")
            return None
        
    text = data_path.read_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ── 5. RAG prompt template ──────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer the question. If the information is not in the context, say you don't know.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])

# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── 7. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    """
    return chain.invoke(question)

# ── 8. Sample questions (50 total) ────────────────────
SAMPLE_QUESTIONS = [
    "What are the three main types of machine learning?",
    "What is overfitting in machine learning?",
    "Explain the bias-variance tradeoff.",
    "How does regularization prevent overfitting?",
    "What is cross-validation?",
    "What is backpropagation?",
    "What are Convolutional Neural Networks primarily used for?",
    "How do LSTM networks address the vanishing gradient problem?",
    "What activation functions are commonly used in neural networks?",
    "What is the role of pooling layers in CNNs?",
    "What is the transformer architecture?",
    "What are word embeddings?",
    "What is transfer learning in NLP?",
    "How does BERT handle language understanding?",
    "What is self-attention in transformers?",
    "What is GPT and how is it trained?",
    "What is instruction tuning?",
    "What is RLHF?",
    "What is chain-of-thought prompting?",
    "What is the context length of GPT-4?",
    "What is Retrieval-Augmented Generation?",
    "What are the main components of a RAG pipeline?",
    "What is dense retrieval?",
    "Why is chunking strategy important in RAG?",
    "What advanced RAG techniques exist beyond basic retrieval?",
    "What are vector databases used for?",
    "What is FAISS?",
    "How do text embeddings capture semantic meaning?",
    "What is HNSW?",
    "What is hybrid search in vector databases?",
    "What is LangChain?",
    "What is LangChain Expression Language (LCEL)?",
    "What is LangGraph?",
    "What memory types does LangChain support?",
    "What are LangChain retrievers?",
    "What is LangSmith?",
    "What information do LangSmith traces capture?",
    "What is the LangSmith Prompt Hub?",
    "How does LangSmith help monitor production LLM applications?",
    "What are LangSmith datasets used for?",
    "What is RAGAS?",
    "How does RAGAS compute faithfulness?",
    "What is answer relevancy in RAGAS?",
    "What is context recall in RAGAS?",
    "What inputs does RAGAS evaluation require?",
    "What is Guardrails AI?",
    "What is PII and why is it important to detect in LLM responses?",
    "What does structured output validation ensure?",
    "What is Constitutional AI?",
    "What are common AI safety concerns with LLMs?",
]

# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    vectorstore = build_vectorstore()
    if not vectorstore:
        return

    chain, retriever = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        try:
            answer = ask(chain, question)
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
            print(f"       A: {answer[:100]}\n")
            
            # Rate limit handling based on provider
            if os.getenv("GROQ_API_KEY"):
                time.sleep(2)  # Groq is fast and has high RPM
            elif os.getenv("GOOGLE_API_KEY"):
                time.sleep(12) # Gemini Free Tier (5-15 RPM)
        except Exception as e:
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Error: {e}")

    print(f"✅ {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")

if __name__ == "__main__":
    main()
