import re
from states.state import State

class SpanFormat:
    
    def __init__(self):
        pass
    
    def __call__(self, state: State) -> State:
        #print('OUTPUT OutputCorrection & INPUT SpanFormat: ', state)
        state.span_signals = []

        for segment_text, segment_signal in zip(state.segmented_text, state.segmented_signals):
            span_signal = []
            
            for category, entity_list in segment_signal.items():
                for entity in entity_list:
                    # Trova tutte le occorrenze di `entity` nel testo
                    if entity and len(entity)>1:
                        for match in re.finditer(re.escape(entity), segment_text):
                            span_signal.append({
                                "type": category,
                                "text": entity,
                                "start": match.start(),
                                "end": match.end()
                            })
                    
            state.span_signals.append(span_signal)
        
        return state