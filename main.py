import os
import json
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.GeminiCostAnalyze import GeminiCostAnalyze

from pipelines.api_pipeline import run_pipeline 
# from pipelines.cpp_pipeline import run_pipeline 

# CONFIG
TOPIC = "election"
INPUT_PATH = f'./data/raw/{TOPIC}_articles_deduplicated.jsonl'
OUTPUT_PATH = f'./data/output/{TOPIC}_articles_annotated.jsonl'
CHECKPOINT_PATH = f'./data/checkpoint/{TOPIC}_checkpoint.json'

PROMPTS_PATH = './config/prompts.yml'
MODEL_CONFIG = "./config/gemini2.0-flash.yml"

def main():
    # Carica i prompt
    with open(PROMPTS_PATH, 'r', encoding='utf-8') as file:
        prompts = yaml.safe_load(file)

    # Carica la configurazione del modello
    with open(MODEL_CONFIG, "r", encoding="utf-8") as f:
        llm_config = yaml.safe_load(f)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        google_api_key=llm_config["api_key"],
        temperature=llm_config["temperature"],
        max_output_tokens=llm_config["max_output_tokens"],
        top_p=llm_config["top_p"],
        top_k=llm_config.get("top_k", None),
    )

    # Crea file se non esistono
    if not os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            pass
    if not os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": 0}, f, ensure_ascii=False, indent=4)

    try:
        run_pipeline(INPUT_PATH, OUTPUT_PATH, CHECKPOINT_PATH, llm, prompts, llm_config)
        #run_pipeline(INPUT_PATH, OUTPUT_PATH, CHECKPOINT_PATH, llm_config, prompts)
        
        os.remove(CHECKPOINT_PATH)
        
        #costAnalyzer non richiesto per i modelli locali
        GeminiCostAnalyze().daily_cost_log()
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")


if __name__ == "__main__":
    main()
