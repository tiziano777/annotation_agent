import json
from datetime import datetime
import pandas as pd

class CostAnalyze:
    """
    Classe per il calcolo costi di servizio API.
    """

    def __init__(self,log_file_path: str = "/home/tiziano/annotation_agent/log/token_cost_log.jsonl"):
        
        self.log_file_path = log_file_path
        
    def daily_cost(self, threshold: float):
        """
        Legge il log e calcola il costo cumulativo del giorno corrente.
        Se supera la soglia `threshold`, solleva un'eccezione.
        """
        today = datetime.now().strftime("%Y%m%d")
        total_cost = 0.0

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("timestamp", "").startswith(today):
                        total_cost += float(entry.get("cost", 0.0))
        except FileNotFoundError:
            return  # Nessun log da analizzare

        if total_cost > threshold:
            raise RuntimeError(f"Daily cost limit exceeded: ${total_cost:.4f} > ${threshold:.4f}")

    def daily_cost_log(self):
        """
        Aggrega i costi giornalieri dal log e salva in `log/cost_log.csv`
        con colonne: Giorno (GG-MM-YYYY), Costo.
        """
        cost_by_day = {}

        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    timestamp = entry.get("timestamp", "")
                    if len(timestamp) >= 8:
                        day_key = timestamp[:8]  # 'YYYYMMDD'
                        cost = float(entry.get("cost", 0.0))
                        cost_by_day[day_key] = cost_by_day.get(day_key, 0.0) + cost
        except FileNotFoundError:
            return

        # Converte a formato DataFrame
        records = []
        for day, cost in sorted(cost_by_day.items()):
            formatted_day = datetime.strptime(day, "%Y%m%d").strftime("%d-%m-%Y")
            records.append({"Giorno": formatted_day, "Costo": round(cost, 8)})

        df = pd.DataFrame(records)
        df.to_csv("log/cost_log.csv", index=False, encoding="utf-8")