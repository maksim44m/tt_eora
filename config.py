import os
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
tg_token = os.getenv('TG_TOKEN')
llm_token = os.getenv('LLM_TOKEN')
llm_url = os.getenv('LLM_URL')


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
LOGS_DIR = DATA_DIR / "logs"

# Создаем папки если их нет
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "log.txt", encoding="utf-8")
    ],
    force=True
)




