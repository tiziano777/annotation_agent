import json
import re
import traceback

from json_repair import repair_json
from states.state import State
from utils.CostLogger import CostLogger
from utils.ErrorHandler import ErrorHandler

class SpanRefiner:
    """
    Nodo LangGraph per raffinare gli span delle entità tramite un LLM.
    Prende in input le annotazioni e chiede al modello di correggerne i bordi.
    """

    def __init__(self, llm, prompts=None):
        """
        Inizializza il nodo SpanRefiner.

        :param llm: Modello LangChain-compatible (es. ChatGoogleGenerativeAI).
        :param prompts: Prompt di sistema da usare per il raffinamento degli span.
        """
        self.llm = llm
        self.prompt = prompts
        self.logger = CostLogger()
        self.error_handler = ErrorHandler()
        self.end_prompt = "\n Output: \n"

    def refine_spans(self, state: State) -> State:
        """
        Processa le annotazioni segmentate per raffinare gli span.
        """
        segmented_text = state.segmented_text
        segmented_signals = state.segmented_signals
        
        refined_signals = []
        total_input_tokens = 0
        total_output_tokens = 0

        refinement_prompt_template = self.prompt.get('refinement_prompt', "")

        for i, (text_segment, signals_segment) in enumerate(zip(segmented_text, segmented_signals)):
            print(f"Raffinamento del segmento {i} con testo: {text_segment[:50]}...")
            if not signals_segment:
                refined_signals.append({})
                continue

            # Prepara il testo per il prompt di raffinamento
            signals_str = json.dumps(signals_segment, ensure_ascii=False, indent=2)
            
            full_refinement_prompt = f"{refinement_prompt_template} {signals_str} \n {self.end_prompt}"

            try:
                response = self.error_handler.invoke_with_retry(llm=self.llm, prompt=full_refinement_prompt)
                
                log = response.usage_metadata
                total_input_tokens += log["input_tokens"]
                total_output_tokens += log["output_tokens"]

                repaired_json_str = self.extract_json(response.content)
                if repaired_json_str:
                    refined_signals.append(repaired_json_str)
                else:
                    print(f"Errore: Il raffinamento del JSON per il segmento {i} ha prodotto un output vuoto. Manteniamo le annotazioni originali per questo segmento.")
                    refined_signals.append(signals_segment)

            except Exception as e:
                print(f"Errore durante il raffinamento degli span per il segmento {i}: {e}")
                traceback.print_exc()
                refined_signals.append(signals_segment)
        
        self.logger(total_input_tokens, total_output_tokens)
        
        state.segmented_signals = refined_signals
        state.input_tokens += total_input_tokens
        state.output_tokens += total_output_tokens
        
        # Imposta refined_once a True dopo il primo passaggio di raffinamento
        state.refined_once = True 

        return state

    def extract_json(self, json_text: str) -> dict:
        """
        Estrae un JSON valido da una stringa di output, usando json_repair per tolleranza agli errori.
        """
        try:
            repaired_text = repair_json(json_text)
            json_match = re.search(r"\{.*?\}", repaired_text, re.DOTALL)
            if json_match is None:
                print(f"Nessun JSON valido trovato nel testo:\n\n{json_text}")
                return {}
            parsed_json = json.loads(json_match.group(0))
            return parsed_json
        except Exception as e:
            print(f"Errore nel parsing/riparazione JSON nel SpanRefiner:\n→ Testo:\n{json_text}\n→ Errore: {e}")
            return {}

    def __call__(self, state: State) -> State:
        """
        Entry-point per LangGraph node.
        """
        ### CUSTOM LOGIC (I set only one step of refinement deterministically) ###
        if state.refined_once:
            step = 2
        else:
            step = 1
        #print(f'INPUT SpanRefiner & OUTPUT di OutputCorrection step {step} : ', state.segmented_signals)
        return self.refine_spans(state)