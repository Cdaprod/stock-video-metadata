# scripts/llm.py

import os
import json

# Monkey-patch OpenAI if needed
try:
    import openai
    from openai._base_client import SyncHttpxClientWrapper
    if tuple(map(int, openai.__version__.split("."))) <= (1, 55, 3):
        class NoProxiesWrapper(SyncHttpxClientWrapper):
            def __init__(self, *args, **kwargs):
                kwargs.pop("proxies", None)
                super().__init__(*args, **kwargs)
        openai._base_client.SyncHttpxClientWrapper = NoProxiesWrapper
    # Import OpenAI error if available for targeted excepts
    try:
        OpenAIRateLimitError = openai.error.RateLimitError
    except Exception:
        OpenAIRateLimitError = Exception
except Exception as e:
    print(f"OpenAI patch failed (may be unnecessary): {e}")
    OpenAIRateLimitError = Exception

# LangChain for OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Optional: local llama-cpp model
LOCAL_LLM_OK = False
llama_llm = None
try:
    from llama_cpp import Llama
    LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH", "models/llama-2-7b-chat.Q4_K_M.gguf")
    llama_llm = Llama(
        model_path=LOCAL_LLM_PATH,
        n_ctx=2048,
        n_threads=8,
        verbose=False,
    )
    LOCAL_LLM_OK = True
except Exception as e:
    print(f"⚠️ Local LLM not available: {e}")

class MetadataLLM:
    def __init__(self, openai_model="gpt-4o", local_llm_ok=LOCAL_LLM_OK):
        self.openai_model = openai_model
        self.local_llm_ok = local_llm_ok
        self.llama_llm = llama_llm  # could allow passing in a custom instance

    def _local_llm_infer(self, prompt, max_tokens=256, temperature=0.3):
        if not self.llama_llm:
            raise RuntimeError("llama-cpp LLM is not loaded")
        resp = self.llama_llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\n", "Output JSON only."]
        )
        # llama.cpp returns a dict with "choices"
        text = resp["choices"][0]["text"]
        return text

    def refine_metadata(self, meta: dict) -> dict:
        prompt_template = PromptTemplate.from_template("""
        Refine this video metadata into a concise, accurate description:
        Caption: {caption}
        YOLO Objects: {yolo_objects}
        OCR Text: {ocr_text}
        SceneType: {scene_type}
        MainAction: {main_action}
        SceneMood: {scene_mood}

        Produce:
        - Description: 15–200 chars, at least 5 words.
        - Keywords: 8–49 unique words separated by commas.
        - Category: one of ["Nature", "Infrastructure", "People", "Business", "Technology", "Culture"].
        - Title: short, under 100 chars.

        Output JSON only.
        """)

        # Defensive defaults
        meta = {k: meta.get(k, "") for k in [
            "AI_Description", "YOLO_Objects", "OCR_Text",
            "SceneType", "MainAction", "SceneMood"
        ]}
        prompt = prompt_template.format(
            caption=meta["AI_Description"],
            yolo_objects=meta["YOLO_Objects"],
            ocr_text=meta.get("OCR_Text", ""),
            scene_type=meta["SceneType"],
            main_action=meta["MainAction"],
            scene_mood=meta["SceneMood"]
        )

        ## Try OpenAI first, catch specific rate/HTTP errors and any exception
        try:
            llm = ChatOpenAI(model=self.openai_model, temperature=0.3)
            response = llm.invoke(prompt)
            return json.loads(response.content)
        except OpenAIRateLimitError as e:
            print(f"⚠️ OpenAI quota/rate limit hit: {e}. Falling back to local LLM.")
        except Exception as e:
            print(f"⚠️ OpenAI or LangChain error: {e}. Falling back to local LLM (if available).")

        # Fallback: Local LLM
        if self.local_llm_ok and self.llama_llm is not None:
            try:
                local_response = self._local_llm_infer(prompt)
                return json.loads(local_response)
            except Exception as e:
                print(f"⚠️ Local LLM failed: {e}")
                return {}
        else:
            print("⚠️ No LLM backend available.")
            return {}