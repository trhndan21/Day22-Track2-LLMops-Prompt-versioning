"""
Step 2 — Prompt Hub & A/B Routing
===================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions through the router → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Environment / imports ────────────────────────────────────────────────
load_dotenv()

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "lab22_llmops")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

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
from langsmith import Client, traceable

# ── 2. Provider Factories ───────────────────────────────────────────────────
def get_llm():
    # 1. Try Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            groq_api_key=groq_key
        )
    # 2. Try Gemini
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            google_api_key=google_key
        )
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )

def get_embeddings():
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    if use_local:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("OPENAI_BASE_URL", None),
    )

# ── 3. Define two prompt templates ──────────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

# Use a unique name for the prompts
PROMPT_V1_NAME = "lab22-concise-v1"
PROMPT_V2_NAME = "lab22-structured-v2"

# ── 4. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """
    Upload both prompt versions to LangSmith Prompt Hub.
    """
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise answers")
        print(f"✅ Pushed V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1: {e}")

    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured answers")
        print(f"✅ Pushed V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2: {e}")

# ── 5. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """
    Download both prompt versions from LangSmith Prompt Hub.
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Using local fallback for '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts

# ── 6. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on the MD5 hash of request_id.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

# ── 7. Build vectorstore ─────────────────────────────────────────────────────
def build_vectorstore():
    data_path = Path("data/knowledge_base.txt")
    if not data_path.exists():
        data_path = Path("../data/knowledge_base.txt")
        if not data_path.exists():
            print(f"Error: knowledge_base.txt not found.")
            return None
    
    embeddings = get_embeddings()
    text = data_path.read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ── 8. Traced A/B query function ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Run the RAG chain using the given prompt version.
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    chain = (prompt | llm | StrOutputParser())
    answer = chain.invoke({"context": context, "question": question})
    
    return {"question": question, "answer": answer, "version": version}

# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

    # Push prompts
    push_prompts_to_hub(client)

    # Pull prompts
    prompts = pull_prompts_from_hub(client)

    # Setup RAG
    vectorstore = build_vectorstore()
    if not vectorstore:
        return
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    # Import SAMPLE_QUESTIONS from Step 1
    import importlib.util
    file_path = Path(__file__).parent / "01_langsmith_rag_pipeline.py"
    spec = importlib.util.spec_from_file_location("step1", str(file_path))
    step1 = importlib.util.module_from_spec(spec)
    sys.modules["step1"] = step1
    spec.loader.exec_module(step1)
    sample_questions = step1.SAMPLE_QUESTIONS

    v1_count = 0
    v2_count = 0

    log_file = "02_ab_routing_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Step 2: A/B Routing Log\n")
        f.write("=" * 60 + "\n\n")

        for i, question in enumerate(sample_questions, 1):
            request_id = f"req-{i:04d}"
            version_key = get_prompt_version(request_id)
            version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
            prompt = prompts[version_key]

            if version_tag == "v1": v1_count += 1
            else: v2_count += 1

            try:
                result = ask_ab(retriever, llm, prompt, question, version_tag)
                log_entry = f"[{i:02d}] [prompt-{version_tag}] {question}\n"
                print(log_entry.strip())
                f.write(log_entry)
                
                # Rate limit handling
                if os.getenv("GROQ_API_KEY"):
                    time.sleep(2)
                elif os.getenv("GOOGLE_API_KEY"):
                    time.sleep(12)
            except Exception as e:
                err_msg = f"[{i:02d}] Error: {e}\n"
                print(err_msg.strip())
                f.write(err_msg)

    print(f"\n✅ A/B routing complete: V1={v1_count}, V2={v2_count}")
    print(f"📄 Log saved to: {log_file}")

if __name__ == "__main__":
    main()
