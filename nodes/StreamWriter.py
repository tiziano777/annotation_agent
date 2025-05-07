import json
from states.state import State

class StreamWriter:
    def __init__(self, output_file):
        self.file = output_file

    def __call__(self, state: State) -> State:
        print('OUTPUT SpanFormat & INPUT Writer: ', state)
        try:
            data_to_write = []  # Lista per accumulare i dati da scrivere
            for title, url,lang, clickbait, text, span in zip( state.title, state.url,state.language, state.clickbait, state.segmented_text, state.span_signals):
                if span != []:  # Controlla se 'signal' non è un dizionario vuoto
                    data_to_write.append({'title':title, 'url':url,'lang':lang, 'clickbait':clickbait, 'text': text, 'span': span })

            # Scrivi tutto in un'unica operazione
            with open(self.file, "a", encoding="utf-8") as f:
                for item in data_to_write:
                    json.dump(item, f)
                    f.write("\n")
            
            return {'error_status': None}
        except:
            print(f'cannot write on file this data: {text}')
            state.error_status=f'cannot write on file this data: {text}'
            return state