import json
import re
import spacy
import traceback
import time
from typing import List, Dict, Any

from json_repair import repair_json

from states.state import State
from utils.GeminiCostLogger import GeminiCostLogger 


def extract_retry_delay_from_error(e) -> float | None:
    """
    Estrae il retry delay da un'eccezione ResourceExhausted o simile.

    Presuppone che l'eccezione `e` abbia un attributo `details`, 
    dove nella posizione 2 (convenzionalmente) è presente un dizionario
    con 'retry_delay' espresso come {'seconds': int}.
    """
    try:
        if hasattr(e, 'code') and e.code == 429:
            # Estrarre campo "retry_delay" se disponibile
            #delay = e.details[2]['retry_delay']['seconds']
            delay = e.details[2].retry_delay.seconds
            print(delay)
            return float(delay)
    except Exception as ex:
        print(f"[extract_retry_delay_from_error] Errore durante l'estrazione del retry delay: {ex}")
    
    return None

def invoke_with_retry(llm, prompt, max_retries=5, retry_count=0):
    """
    Richiama `llm.invoke(prompt)` gestendo automaticamente rate-limit con retry ricorsivo.
    
    Se il codice d'errore è 429, estrae `retry_delay` e attende.
    In caso contrario, stampa errore ed esce.

    :param llm: Oggetto LLM (LangChain-compatible).
    :param prompt: Prompt da inviare.
    :param max_retries: Numero massimo di retry.
    :param retry_count: Contatore di retry (usato internamente per ricorsione).
    :return: Risultato di `llm.invoke(prompt)`.
    """
    try:
        return llm.invoke(prompt)

    except Exception as e:
        if hasattr(e, "code") and e.code == 429:
            delay = extract_retry_delay_from_error(e)
            if delay is not None:
                print(f"[invoke_with_retry] Rate limit: attendo {delay:.2f} secondi (retry #{retry_count + 1})")
                time.sleep(delay)
                if retry_count < max_retries:
                    return invoke_with_retry(llm, prompt, max_retries=max_retries, retry_count=retry_count + 1)
                else:
                    print("[invoke_with_retry] Numero massimo di retry superato. Esco.")
                    exit(1)
            else:
                print("[invoke_with_retry] Retry delay non trovato nell'errore 429. Esco.")
                exit(1)
        else:
            print(f"[invoke_with_retry] Errore non gestito: {e}")
            traceback.print_exc()
            exit(1)

# Caricamento modelli SpaCy
en_nlp = spacy.load("en_core_web_sm")
it_nlp = spacy.load("it_core_news_sm")

class Annotator:
    """
    Nodo LangGraph compatibile con modelli LLM API-based (es. Gemini via LangChain).
    """

    def __init__(self, llm, input_context, prompts=None):
        """
        :param llm: Modello LangChain-compatible (es. ChatGoogleGenerativeAI).
        :param input_context: Numero massimo di token di contesto.
        :param prompts: Prompt di sistema da usare per l'annotazione.
        """
        self.llm = llm
        self.system_prompts = prompts
        self.input_context = input_context
        self.logger = GeminiCostLogger()  # Istanza per logging dei token e costi
        self.end_prompt = "\n Output: \n"

    def annotate(self, state: State):
        text = state.text
        language = state.language
        title = state.title or ""

        sentences = self.process_text(language, text)
        signals = []
        texts = []

        total_input_tokens = 0
        total_output_tokens = 0
        clickbait_score = 0

        ####### CALL CUSTOM LOGIC #######
        
        # Clickbait detection
        clickbait_prompt = self.system_prompts.get('clickbait_prompt', "")
        try:
            full_cb_prompt = clickbait_prompt + title
            raw_clickbait = invoke_with_retry(self.llm, full_cb_prompt)

            
            log = raw_clickbait.usage_metadata
            total_input_tokens += log["input_tokens"]
            total_output_tokens += log["output_tokens"]

            json_clickbait = self.extract_json(raw_clickbait.content)
            clickbait_score = int(json_clickbait.get('clickbait', 0))
            
        except Exception as e:
            print(f"Errore nel clickbait parsing: {e}")
            traceback.print_exc()
        
        # Segment-level annotation
        annotation_prompts = self.system_prompts.get('election_optimized_strategy')
        gate_prompt = self.system_prompts.get('gate_prompt', "")

        for s in sentences:
            cumulative_annotations = []
            
            if s.strip():
                # ⛔️ Step 1: GATE STEP To avoid unusefull calls (Skip if score < 3)
                
                try:
                    gate_response = invoke_with_retry(self.llm, gate_prompt + s + self.end_prompt)
                    log = gate_response.usage_metadata
                    total_input_tokens += log["input_tokens"]
                    total_output_tokens += log["output_tokens"]

                    json_gate = self.extract_json(gate_response.content)
                    gate_score = float(json_gate.get('score', 0))
                except Exception as e:
                    print(f"Errore nel gate scoring: {e}")
                    continue  # Skip segment if error in gate or malformed output

                if gate_score < 3:
                    print(f"gate score too low {gate_score}, skipping segment \n")
                    print(f" \n Segmento skipped: {s} \n\n")
                    continue  # Skip segment if gate score is below threshold

                texts.append(s)

                # ✅ Step 2: Run segment-level annotations (only if gate passed)
                for prompt_name, prompt_template in annotation_prompts.items():
                    try:
                        full_prompt = prompt_template + s + self.end_prompt
                        response = self.llm.invoke(full_prompt)
                        
                        log = response.usage_metadata
                        total_input_tokens += log["input_tokens"]
                        total_output_tokens += log["output_tokens"]

                        parsed = self.extract_json(response.content)
                        cumulative_annotations.append(parsed)

                    except Exception as e:
                        print(f"Errore nella annotazione con prompt {prompt_name}: {e}")
                        traceback.print_exc()
                        cumulative_annotations.append({prompt_name: []})

                cumulative_annotations = self.flatten_dict_list(cumulative_annotations)
                signals.append(cumulative_annotations)

        self.logger(total_input_tokens,total_output_tokens)
        
        return {
            'clickbait': clickbait_score,
            'segmented_text': texts,
            'segmented_signals': signals,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
        } 
    
    def extract_json(self, json_text: str) -> dict:
        """
        Estrae un JSON valido da una stringa di output. Usa json_repair per tolleranza agli errori.
        """
        try:
            json_text = repair_json(json_text)
            json_match = re.search(r"\{.*?\}", json_text, re.DOTALL)
            if json_match is None:
                print(f"Nessun JSON trovato nel testo:\n\n{json_text}")
                return {}
            repaired_json = repair_json(json_match.group(0))
            parsed_json = json.loads(repaired_json)    
            
            return parsed_json

        except Exception as e:
            print(f"Errore nel parsing JSON:\n→ Testo:\n{json_text}\n→ Errore: {e}")
            return {}

    def process_text(self, lang: str, text: str) -> List[str]:
        """
        Segmenta il testo in chunk semantici, adattando la lunghezza in base ai token effettivi.
        """
        nlp = it_nlp if lang == 'Italian' else en_nlp
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        max_tokens = self.input_context 
        chunks = []
        current_chunk = []

        for sentence in sentences:
            test_chunk = current_chunk + [sentence]
            test_text = " ".join(test_chunk)
            token_count =  max(1, round(len(test_text) / 4.0))
            if token_count <= max_tokens:
                current_chunk.append(sentence)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]

        if current_chunk:
            chunks.append(" ".join(current_chunk))


        return chunks

    def flatten_dict_list(self,dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Appiattisce una lista di dizionari in un unico dizionario,
        aggregando tutte le chiavi. In caso di chiavi duplicate,
        il valore viene sovrascritto (ultimo vince).

        Parameters:
            dict_list (List[Dict[str, Any]]): Lista di dizionari.

        Returns:
            Dict[str, Any]: Dizionario appiattito.
        """
        flat_dict = {}
        for d in dict_list:
            if not isinstance(d, dict):
                raise TypeError(f"Elemento non valido nella lista: atteso dict, trovato {type(d)}")
            flat_dict.update(d)  # sovrascrive chiavi duplicate con l'ultimo valore
        return flat_dict

    def __call__(self, state: State):
        """
        Entry-point per LangGraph node.
        """
        #print('INPUT Annotator: ', state)
        return self.annotate(state)
