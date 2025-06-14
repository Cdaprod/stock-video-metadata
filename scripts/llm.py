# scripts/llm.py

import os
import json

# --- OpenAI monkey-patch for legacy proxy issues ---
try:
    import openai
    from openai._base_client import SyncHttpxClientWrapper
    if tuple(map(int, openai.__version__.split("."))) <= (1, 55, 3):
        class NoProxiesWrapper(SyncHttpxClientWrapper):
            def __init__(self, *args, **kwargs):
                kwargs.pop("proxies", None)
                super().__init__(*args, **kwargs)
        openai._base_client.SyncHttpxClientWrapper = NoProxiesWrapper
    # Import OpenAI error for fine-grained excepts
    try:
        OpenAIRateLimitError = openai.error.RateLimitError
    except Exception:
        OpenAIRateLimitError = Exception
except Exception as e:
    print(f"\n[llm.py] ⚠️ OpenAI monkey-patch failed (may be unnecessary): {e}\n", flush=True)
    OpenAIRateLimitError = Exception

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# --- Load local llama-cpp model if available ---
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
    print(f"[llm.py] 🦙 Local Llama model loaded: {LOCAL_LLM_PATH}\n", flush=True)
except Exception as e:
    print(f"[llm.py] ⚠️ Local LLM not available: {e}\n", flush=True)

class MetadataLLM:
    def __init__(self, openai_model="gpt-4o", local_llm_ok=LOCAL_LLM_OK):
        self.openai_model = openai_model
        self.local_llm_ok = local_llm_ok
        self.llama_llm = llama_llm

    def _local_llm_infer(self, prompt, max_tokens=256, temperature=0.3):
        print(f"\n[llm.py] 🦙 Invoking local Llama.cpp LLM...", flush=True)
        if not self.llama_llm:
            raise RuntimeError("llama-cpp LLM is not loaded")
        resp = self.llama_llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\n\n", "Output JSON only."]
        )
        text = resp["choices"][0]["text"]
        print(f"[llm.py] 🦙 Local LLM raw response: {text[:100]}...", flush=True)
        return text

    def refine_metadata(self, meta: dict) -> dict:
        print(f"\n[llm.py] 📝 Starting metadata refinement...", flush=True)
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

        # Defensive defaults for safety
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

        # ---- Try OpenAI LLM ----
        try:
            print(f"[llm.py] 🤖 Invoking OpenAI ({self.openai_model})...", flush=True)
            llm = ChatOpenAI(model=self.openai_model, temperature=0.3)
            response = llm.invoke(prompt)
            print(f"[llm.py] ✅ OpenAI LLM responded", flush=True)
            result = json.loads(response.content)
            print(f"[llm.py] 📝 OpenAI refined metadata: {result}", flush=True)
            return result
        except OpenAIRateLimitError as e:
            print(f"[llm.py] ⏳ OpenAI quota/rate limit hit: {e}. Falling back to local LLM.", flush=True)
        except Exception as e:
            print(f"[llm.py] ⚠️ OpenAI or LangChain error: {e}. Falling back to local LLM (if available).", flush=True)

        # ---- Fallback: Local Llama.cpp ----
        if self.local_llm_ok and self.llama_llm is not None:
            try:
                local_response = self._local_llm_infer(prompt)
                result = json.loads(local_response)
                print(f"[llm.py] 📝 Local LLM refined metadata: {result}", flush=True)
                return result
            except Exception as e:
                print(f"[llm.py] ❌ Local LLM failed: {e}\n", flush=True)
                return {}
        else:
            print("[llm.py] 🚫 No LLM backend available.", flush=True)
            return {}

    # --- (OPTIONAL) LangChain Expression Language pattern ---
    # To use LCEL's RunnableWithFallbacks, comment out the above .refine_metadata
    # and use a chain like this (uncomment to try!):
    #
    # from langchain_core.runnables import RunnableLambda
    # from langchain_core.runnables import RunnableWithFallbacks
    #
    # def openai_refine(prompt: str) -> str:
    #     llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    #     return llm.invoke(prompt).content
    #
    # def llama_refine(prompt: str) -> str:
    #     if llama_llm is None:
    #         raise RuntimeError("Local Llama LLM not loaded")
    #     resp = llama_llm(prompt, max_tokens=256, temperature=0.3)
    #     return resp["choices"][0]["text"]
    #
    # main_runnable = RunnableLambda(openai_refine)
    # fallback_runnables = []
    # if LOCAL_LLM_OK:
    #     fallback_runnables = [RunnableLambda(llama_refine)]
    # meta_refiner = main_runnable.with_fallbacks(
    #     fallback_runnables, exceptions_to_handle=(Exception,)
    # )
    #
    # def refine_metadata(self, meta: dict) -> dict:
    #     prompt = ... # build prompt as above
    #     text = meta_refiner.invoke(prompt)
    #     return json.loads(text)

# End of llm.py