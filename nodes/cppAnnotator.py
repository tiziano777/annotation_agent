import json
import re
import spacy
from states.state import State
from transformers import AutoTokenizer
from typing import List
from json_repair import repair_json

# Caricamento dei modelli linguistici per l'analisi del testo
en_nlp = spacy.load("en_core_web_sm")
it_nlp = spacy.load("it_core_news_sm")

class Annotator:
    """
    Wrapper per un modello LLM compilato con llama.cpp, utilizzabile come nodo di elaborazione in LangGraph.
    """

    def __init__(self, llm, input_context, prompts=None, context_padding_length=512):
        """
        Inizializza il modello API.

        :param llm: Modello quantizzato compilato con llama.cpp (GGUF).
        :param input_context: Numero massimo di token per il contesto di input.
        :param prompts: Prompt di sistema da utilizzare per l'annotazione.
        :param budget_for_user_input: percentuale massima contesto delle frasi in input.
        """
        self.llm = llm
        self.system_prompts = prompts
        self.input_context = input_context
        self.end_prompt = "\n Output JSON Syntax: \n"
        self.context_padding_length = context_padding_length

        # Tokenizer di Gemma 3
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it", trust_remote_code=True)

    def annotate(self, state: State):
        """
        Genera un'annotazione per il testo di input utilizzando il modello
        e converte il risultato in un JSON strutturato.

        :param state: Stato contenente il testo da elaborare.
        :return: Un dizionario Python con l'output JSON.
        """
        text = state.text
        sentences = self.process_text(state.language, text)
        signals = []
        texts = []

        ### CUSTOM LOGIC ###
        
        # Elaborazione del titolo per il clickbait
        if state.title:
            clickbait_prompt = self.system_prompts['clickbait_prompt']
            json_clickbait = self.extract_json(self.llm.invoke(clickbait_prompt + state.title))

        for s in sentences:
            cumulative_annotations = []
            if s:
                texts.append(s)
                for prompt_name, prompt_content in self.system_prompts.get('annotation_prompts', []).items():
                    response = self.llm.invoke(prompt_content + s + self.end_prompt)
                    
                    print('Prompt: ', prompt_name)
                    print('INPUT: ', s)
                    print('\n OUTPUT', response)
                    print('----------------------------------')
                    

                cumulative_annotations.append(self.extract_json(response))
            signals.append(cumulative_annotations)

        #print('OUTPUT Title Annotator: ', json_clickbait)
        #print('OUTPUT Text Annotator: ', texts)
        #print('OUTPUT Signal Annotator: ', signals)
        return {'clickbait': int(json_clickbait['clickbait']), 'segmented_text': texts, 'segmented_signals': signals}

    def extract_json(self, text: str) -> dict:
        """
        Estrae e corregge l'output JSON generato dal modello utilizzando la libreria `json_repair`:
        """
        try:
            repaired_json = repair_json(json_text)
            json_match = re.search(r"\{.*?\}", text, re.DOTALL)
            if not json_match:
                repaired_json = repair_json(json_text)
                raise ValueError("Nessun JSON valido trovato nel testo: ",text)

            json_text = json_match.group(0)

            # Correggi il JSON utilizzando la libreria json_repair
            repaired_json = repair_json(json_text)

            if repaired_json is None:
                raise ValueError("Impossibile riparare il JSON.")

            parsed_json = json.loads(repaired_json)
            return parsed_json

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Errore nel parsing JSON:\n\n[ text: \n\n {text} \n\n error: {e} ]")
            return {}

    def process_text(self, lang: str, text: str) -> List[str]:
        """
        Segmenta il testo in chunk semantici concatenando frasi fino a rientrare nel token budget.

        :param lang: Lingua del testo ('English' o 'Italian').
        :param text: Testo da elaborare.
        :return: Lista di chunk semantici.
        """
        # Carica il sentencer per la lingua specificata
        nlp = it_nlp if lang == 'Italian' else en_nlp
        doc = nlp(text)

        # Segmenta il testo in frasi
        sentences = [sent.text.strip() for sent in doc.sents]

        # Calcola il massimo numero di token dei prompt di sistema
        prompt_token_lengths = [
            len(self.tokenizer.tokenize(p)) for p in self.system_prompts.get('annotation_prompts', {}).values()
        ]
        
        print("Prompt token lengths: ", prompt_token_lengths)
        max_prompt_tokens = max(prompt_token_lengths) if prompt_token_lengths else 0
        token_budget = int(self.input_context - max_prompt_tokens -self.context_padding_length)

        print("Max Token prompt budget: ", max_prompt_tokens)
        print("Token budget: ", token_budget)

        # Concatenazione di frasi fino al raggiungimento del token budget
        chunks = []
        current_chunk = []
        current_token_count = 0

        for sentence in sentences:
            sentence_token_count = len(self.tokenizer.tokenize(sentence))

            # Se l'aggiunta della frase non supera il token budget, aggiungi la frase al chunk corrente
            if current_token_count + sentence_token_count <= token_budget:
                current_chunk.append(sentence)
                current_token_count += sentence_token_count
            else:
                # Se supera il token budget, salva il chunk corrente e inizia un nuovo chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_token_count = sentence_token_count

        # Aggiungi l'ultimo chunk se non è vuoto
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Stampa i chunk risultanti
        
        for i, chunk in enumerate(chunks):
            print(f"CHUNK {i+1} (lunghezza {len(self.tokenizer.tokenize(chunk))})")
        
        
        return chunks

    def __call__(self, text: State):
        """
        Metodo principale per elaborare il testo di input.

        :param text: Il testo da elaborare.
        :return: Un dizionario Python con l'output JSON.
        """
        print('INPUT Annotator: ', text)
        return self.annotate(text)
