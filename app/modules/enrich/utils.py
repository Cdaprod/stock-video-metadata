# app/modules/enrich/utils.py
import os
from dotenv import load_dotenv
from openai._base_client import SyncHttpxClientWrapper
load_dotenv()

def patch_openai_no_proxy():
    try:
        if tuple(map(int, openai.__version__.split("."))) <= (1,55,3):
            class NoProxies(SyncHttpxClientWrapper):
                def __init__(self,*a,**kw):
                    kw.pop("proxies",None)
                    super().__init__(*a,**kw)
            openai._base_client.SyncHttpxClientWrapper = NoProxies
    except Exception:
        pass

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")