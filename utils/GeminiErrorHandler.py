
import traceback
import time

class GeminiErrorHandler:
    
    def __init__(self):
        pass
    
    def extract_retry_delay_from_error(self, e) -> float | None:
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

    def invoke_with_retry(self, llm, prompt, max_retries=5, retry_count=0):
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
                delay = self.extract_retry_delay_from_error(e)
                if delay is not None:
                    print(f"[invoke_with_retry] Rate limit: attendo {delay:.2f} secondi (retry #{retry_count + 1})")
                    time.sleep(delay)
                    if retry_count < max_retries:
                        return self.invoke_with_retry(llm, prompt, max_retries=max_retries, retry_count=retry_count + 1)
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
