import re
from rapidfuzz import fuzz, process
from states.state import State

class OutputCorrection():
    def __init__(self, similarity_threshold=79):
        self.similarity_threshold = similarity_threshold
    
    def extract_ngram_candidates(self, text, entity_length):
        """
        Estrae sottostringhe di lunghezza `entity_length` dal testo.
        """
        words = text.split()
        ngrams = [' '.join(words[i:i + entity_length]) for i in range(len(words) - entity_length + 1)]
        return ngrams

    def clean_entity(self,entity):
        """
        Rimuove la punteggiatura solo all'inizio e alla fine dell'entità.
        """
        # Rimuove la punteggiatura solo all'inizio della stringa
        entity = re.sub(r'^[^\w\d]+', '', entity)
        
        # Rimuove la punteggiatura solo alla fine della stringa
        entity = re.sub(r'[^\w\d]+$', '', entity)
    
        return entity

    def clean_annotation(self, text,signals):
        """
        Corregge le entità in `signals` in base al testo `text`, eliminando duplicati e ripulendo i caratteri speciali.
        """
        text_words = set(str(text).split())  # Parole nel testo
        corrected_signals = {}
        
        for category, entities in signals.items(): # for each category key

            unique_entities = set()
                
            for entity in entities: # for each entity value

                entity_words = set(str(entity).split())  # Parole dell'entità
                
                # Controllo se tutte le parole dell'entità sono presenti nel testo
                if entity_words.issubset(text_words):
                    unique_entities.add(self.clean_entity(entity))
                else:
                    entity_length = len(str(entity).split())  # Conta le parole dell'entità
                    candidates = self.extract_ngram_candidates(text, entity_length)
                    
                    #caso in cui una entita sia piu grande di un testo chunked
                    if len(str(entity))>len(str(text)) or candidates==[]:
                        continue
                    
                    # Trova la migliore corrispondenza nel testo
                    best_match, score, _ = process.extractOne(str(entity), candidates, scorer=fuzz.ratio)
                    if best_match and score >= self.similarity_threshold:
                        # Normalizza la corrispondenza finale
                        best_match = self.clean_entity(best_match)  # Pulisce la corrispondenza finale
                        unique_entities.add(best_match)  # Sostituisci con il match migliore
            
            if len(unique_entities) == 0:
                corrected_signals[category] = []
            else:
                corrected_signals[category] = list(unique_entities)
        
        return corrected_signals
    
    def __call__(self, state:State):
        print('\nOUTPUT ANNOTATOR & INPUT OutputCorrector: \n', state.segmented_text[:50], "\n",state.segmented_signals)
        signals=[]
        for text,signal in zip(state.segmented_text, state.segmented_signals):
            corrected_signal_item=self.clean_annotation(text,signal)
            signals.append(corrected_signal_item)

        print('\nOUTPUT OutputCorrector: \n', state.segmented_text[:50], "\n",state.segmented_signals)
        state.refined_once = True 
        return {'segmented_signals': signals}
