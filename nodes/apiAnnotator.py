import json
import re
import spacy
import traceback
from typing import List, Dict, Any

from json_repair import repair_json

from states.state import State
from utils.GeminiCostLogger import GeminiCostLogger 

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
        self.end_prompt = "\n Output JSON Syntax: \n"

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

        # Clickbait detection
        clickbait_prompt = self.system_prompts.get('clickbait_prompt', "")
        try:
            full_cb_prompt = clickbait_prompt + title
            raw_clickbait = self.llm.invoke(full_cb_prompt)
            
            log = raw_clickbait.usage_metadata
            total_input_tokens += log["input_tokens"]
            total_output_tokens += log["output_tokens"]

            json_clickbait = self.extract_json(raw_clickbait.content)
            clickbait_score = int(json_clickbait.get('clickbait', 0))
            
        except Exception as e:
            print(f"Errore nel clickbait parsing: {e}")
        
        # Segment-level annotation
        annotation_prompts = self.system_prompts.get('annotation_prompts', {})
        for s in sentences:
            cumulative_annotations = []
            if s.strip():
                texts.append(s)
                for prompt_name, prompt_template in annotation_prompts.items():
                    try:
                        full_prompt = prompt_template + s + self.end_prompt
                        response = self.llm.invoke(full_prompt)
                        
                        log = response.usage_metadata
                        total_input_tokens += log["input_tokens"]
                        total_output_tokens += log["output_tokens"]

                        parsed = self.extract_json(response.content)
                        cumulative_annotations.append(parsed)
                        #print(f"[{prompt_name}] Input: {s}\n→ Output: {parsed}\n")
                        
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

    
    
    def extract_json(self, text: str) -> dict:
        """
        Estrae un JSON valido da una stringa di output. Usa json_repair per tolleranza agli errori.
        """
        try:
            json_match = re.search(r"\{.*\}", text.strip(), re.DOTALL)
            json_text = json_match.group(0)
            repaired_json = repair_json(json_text)
            return json.loads(repaired_json)

        except Exception as e:
            print(f"Errore nel parsing JSON:\n→ Testo:\n{text}\n→ Errore: {e}")
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
            token_count = self.logger.count_tokens(test_text)
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
        print('INPUT Annotator: ', state)
        return self.annotate(state)
