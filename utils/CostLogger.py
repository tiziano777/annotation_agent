from datetime import datetime
import json

class CostLogger:
    INPUT_COST_PER_MILLION = 0.10     # Gemini Flash input
    OUTPUT_COST_PER_MILLION = 0.60    # Gemini Flash output

    def __init__(self,
                 input_cost=INPUT_COST_PER_MILLION,
                 output_cost=OUTPUT_COST_PER_MILLION,
                 log_file_path: str = "/home/tiziano/annotation_agent/log/token_cost_log.jsonl"):
        
        self.log_file_path = log_file_path
        self.INPUT_COST_PER_MILLION = input_cost
        self.OUTPUT_COST_PER_MILLION = output_cost

    def compute_cost(self, num_tokens: int, typ: int) -> float:
        rate = self.INPUT_COST_PER_MILLION if typ == 0 else self.OUTPUT_COST_PER_MILLION
        return round((num_tokens / 1_000_000) * rate, 8)

    def _log(self, entry: dict):
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def __call__(self, input_tokens: int, output_tokens: int) -> dict:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        entry = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": self.compute_cost(input_tokens, 0),
            "output_cost": self.compute_cost(output_tokens, 1),
            "timestamp": timestamp,
        }
        self._log(entry)
        return entry
