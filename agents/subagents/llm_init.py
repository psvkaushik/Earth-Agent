from google.adk.models.lite_llm import LiteLlm
import base64
import logging
import os

from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

llm=LiteLlm(model='openai/EVE-Instruct', base_url="http://localhost:18000/v1", api_key="EMPTY", temperature=0)