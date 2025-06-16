# scripts/llm.py

import os
import json
import requests

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
    try:
        OpenAIRateLimitError = openai.error.RateLimitError
    except Exception:
        OpenAIRateLimitError = Exception
except Exception as e:
    print(f"\n[llm.py] ⚠️ OpenAI monkey-patch failed: {e}\n", flush=True)
    OpenAIRateLimitError = Exception

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# --- Llama.cpp HTTP fallback settings ---
LLAMA_SERVER_URL = os.getenv("LLAMA_CPP_URL", "http://127.0.0.1:8000")

class MetadataLLM:
    def __init__(self, openai_model="gpt-4o", llama_server_url=LLAMA_SERVER_URL):
        self.openai_model = openai_model
        self.llama_server_url = llama_server_url

    def _llama_cpp_http_infer(self, prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> str:
        """
        Calls a running llama.cpp server for inference via HTTP.
        Returns the raw text output (expected to be JSON).
        """
        print(f"\n[llm.py] 🦙 Invoking llama.cpp HTTP API at {self.llama_server_url}/completion ...", flush=True)
        try:
            resp = requests.post(
                f"{self.llama_server_url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "temperature": temperature
                },
                timeout=120
            )
            resp.raise_for_status()
            text = resp.json().get("content", "")
            print(f"[llm.py] 🦙 llama.cpp HTTP raw response: {text[:100]}...", flush=True)
            return text
        except Exception as e:
            print(f"[llm.py] 🚫 llama.cpp HTTP error: {e}\n", flush=True)
            return ""

        def refine_metadata(self, meta: dict, max_attempts=1) -> dict:
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
            - Filename
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

            # Try OpenAI first
            for attempt in range(max_attempts):
                try:
                    print(f"[llm.py] 🤖 Invoking OpenAI ({self.openai_model})...", flush=True)
                    llm = ChatOpenAI(model=self.openai_model, temperature=0.3)
                    response = llm.invoke(prompt)
                    print(f"[llm.py] ✅ OpenAI LLM responded", flush=True)
                    result = json.loads(response.content)
                    print(f"[llm.py] 📝 OpenAI refined metadata: {result}", flush=True)
                    return result
                except OpenAIRateLimitError as e:
                    print(f"[llm.py] ⏳ OpenAI quota/rate limit hit: {e}. Falling back to llama.cpp HTTP.", flush=True)
                    break  # Don't retry OpenAI for quota issues
                except Exception as e:
                    print(f"[llm.py] ⚠️ OpenAI or LangChain error: {e}.", flush=True)
                    if attempt < max_attempts - 1:
                        print(f"[llm.py] Retrying OpenAI ({attempt+1}/{max_attempts})...", flush=True)
                        continue
                    else:
                        print(f"[llm.py] OpenAI attempts exhausted, falling back to llama.cpp HTTP.", flush=True)

            # Fallback: llama.cpp HTTP API (single attempt, never loop here)
            text = self._llama_cpp_http_infer(prompt)
            if not text.strip():
                print("[llm.py] 🚨 llama.cpp HTTP returned empty response!", flush=True)
                return {"error": "llama.cpp HTTP empty response"}
            # Heuristic: if llama.cpp response contains "Example output" or "Please provide"
            if "Example" in text or "Please provide" in text:
                print("[llm.py] 🚨 llama.cpp returned a template/example response. Not usable.", flush=True)
                return {"error": "llama.cpp HTTP returned example/template"}
            try:
                result = json.loads(text)
                print(f"[llm.py] 📝 llama.cpp HTTP refined metadata: {result}", flush=True)
                return result
            except Exception as e:
                print(f"[llm.py] ❌ llama.cpp HTTP JSON decode failed: {e}\nRaw output: {text}\n", flush=True)
                return {"error": "llama.cpp HTTP JSON decode failed", "raw_output": text[:500]}

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