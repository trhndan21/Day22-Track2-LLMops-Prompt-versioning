import os
from dotenv import load_dotenv

load_dotenv()

def check_config():
    """
    Verify that all required environment variables are set.
    """
    required = [
        "LANGSMITH_API_KEY",
        "OPENAI_API_KEY"
    ]
    
    missing = [r for r in required if not os.getenv(r)]
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
        
    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {os.getenv('LANGSMITH_PROJECT', 'lab22_llmops')}")
    print(f"   OpenAI endpoint   : {os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')}")
    print(f"   Default LLM model : {os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')}")
    print(f"   Embedding model   : {os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')}")
    
    if os.getenv("OPENAI_BASE_URL") == "https://api.openai.com/v1" or not os.getenv("OPENAI_BASE_URL"):
        print("\n⚠️  Note: You are using the default OpenAI endpoint. If you encounter RateLimitError (429), please check your billing at https://platform.openai.com/account/billing")
    
    return True

if __name__ == "__main__":
    check_config()
