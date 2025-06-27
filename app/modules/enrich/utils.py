# app/modules/enrich/utils.py
import os
from dotenv import load_dotenv
# Only import openai if available (to allow utils to work for llama-only setups)
try:
    import openai
    from openai._base_client import SyncHttpxClientWrapper
except ImportError:
    openai = None
load_dotenv()

def patch_openai_no_proxy():
    if openai is None:
        return
    try:
        if tuple(map(int, openai.__version__.split("."))) <= (1,55,3):
            class NoProxies(SyncHttpxClientWrapper):
                def __init__(self,*a,**kw):
                    kw.pop("proxies",None)
                    super().__init__(*a,**kw)
            openai._base_client.SyncHttpxClientWrapper = NoProxies
    except Exception:
        pass

# Always load keys/paths from environment (or .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def get_llama_cpp_model_path() -> str:
    # This can be set in .env, Compose, or will default to your local model
    return os.getenv("LLAMA_MODEL_PATH", r"B:\Models\llama-2-7b-chat.Q4_K_M.gguf")

def get_llama_cpp_server_url() -> str:
    # Default is localhost; change via .env if running server elsewhere
    return os.getenv("LLAMA_CPP_URL", "http://127.0.0.1:8000")