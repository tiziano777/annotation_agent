import json
from datetime import datetime
from typing import Dict
import pandas as pd
from transformers import T5TokenizerFast

#tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-base")
#import google.generativeai as genai

class GeminiCostLogger:
    """
    Classe per il conteggio token e il calcolo costi su Gemini 2.0 Flash API.
    """

    # Costi per milione di token
    INPUT_COST_PER_MILLION = 0.10     # USD / 1M token (input)
    OUTPUT_COST_PER_MILLION = 0.60    # USD / 1M token (output)

    def __init__(self,
                 input_cost=INPUT_COST_PER_MILLION,
                 output_cost=OUTPUT_COST_PER_MILLION,
                 log_file_path: str = "/home/tiziano/GDELT_scraping/annotation_agent/log/token_cost_log.jsonl",
                 tokenizer_model_path: str = "google/flan-t5-base"):
        
        self.log_file_path = log_file_path
        self.INPUT_COST_PER_MILLION = input_cost
        self.OUTPUT_COST_PER_MILLION = output_cost
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_model_path)

    def count_tokens(self, text: str) -> int:
        """
        Conteggio realistico dei token via SentencePiece (subword).
        """
        return len(self.tokenizer.encode(text))

    def compute_cost(self, num_tokens: int, typ: int) -> float:
        """
        Calcola il costo in USD in base al numero di token.
        typ = 0 → input, typ = 1 → output
        """
        if typ == 0:
            rate = self.INPUT_COST_PER_MILLION
        elif typ == 1:
            rate = self.OUTPUT_COST_PER_MILLION
        else:
            raise ValueError("Invalid type: must be 0 (input) or 1 (output).")

        return round((num_tokens / 1_000_000) * rate, 8)

    ### LOGGING FUNCTIONS

    def _log(self, entry: Dict):
        """
        Salva il log in formato JSON Lines.
        """
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def __call__(self, input_tokens: str, output_tokens: int) -> Dict:
        """
        Calcola token, costo e logga l'invocazione con timestamp.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        entry = {
            "input_token": input_tokens,
            "output_token": output_tokens,
            "input_cost": self.compute_cost(input_tokens, 0),
            "output_cost": self.compute_cost(output_tokens, 1),
            "timestamp": timestamp,
        }

        self._log(entry)
        return entry


