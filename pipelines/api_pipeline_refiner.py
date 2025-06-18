import json
import os
import time

from langgraph.graph import StateGraph, START, END

from states.state import State 

from nodes.apiAnnotator import Annotator
from nodes.OutputCorrection import OutputCorrection
from nodes.SpanFormat import SpanFormat
from nodes.StreamWriter import StreamWriter
from nodes.SpanRefine import SpanRefiner 

from utils.CostAnalyze import CostAnalyze
import traceback


# Funzione di routing che decide il prossimo passo DOPO OutputCorrection
def route_after_correction(state: State) -> str:
    """
    Decide il prossimo nodo in base al flag 'refined_once'.
    - Se refined_once è False, significa che dobbiamo ancora raffinare: vai a SpanRefiner.
    - Se refined_once è True, significa che il raffinamento è stato fatto: vai a SpanFormat.
    """
    if not state.refined_once:
        print("Routing: refined_once è False, andando a SpanRefiner per il raffinamento.")
        return "refine"
    else:
        print("Routing: refined_once è True, andando a SpanFormat (terminato ciclo di raffinamento).")
        return "continue_to_span_format"


def create_pipeline(annotator, output_correction, span_refiner, span_format, writer):
    ### NER annotation pipeline ###
    
    workflow = StateGraph(State)
    workflow.add_node("annotator_node", annotator)
    workflow.add_node("correction_node", output_correction) # Questo nodo sarà usato per entrambi i passaggi di correzione
    workflow.add_node("span_refiner_node", span_refiner)    # Nodo di raffinamento LLM
    workflow.add_node("span_node", span_format)
    workflow.add_node("writer_node", writer)
    
    # Definisci il flusso iniziale
    workflow.add_edge(START, "annotator_node")
    workflow.add_edge("annotator_node", "correction_node")
    
    # Bordo condizionale che parte da 'correction_node'
    workflow.add_conditional_edges(
        "correction_node",        # Il nodo da cui parte il bordo condizionale
        route_after_correction,   # La funzione che decide il percorso
        {
            "refine": "span_refiner_node",         # Se refined_once è False, vai a SpanRefiner
            "continue_to_span_format": "span_node", # Se refined_once è True, vai a SpanFormat
        },
    )

    # Questo è il bordo cruciale che fa tornare il flusso a 'correction_node'
    # dopo che SpanRefiner ha eseguito il suo lavoro e impostato refined_once a True
    workflow.add_edge("span_refiner_node", "correction_node") 
    
    # Bordo finale per la scrittura (raggiunto solo dopo che il ciclo è terminato e si è passato a span_node)
    workflow.add_edge("span_node", "writer_node")
    workflow.add_edge("writer_node", END)
    
    return workflow.compile()


def run_pipeline(input_path, output_path, checkpoint_path, api_llm, prompts, llm_config):
    annotator = Annotator(llm=api_llm, prompts=prompts, input_context=llm_config['n_ctx'])
    correction = OutputCorrection(similarity_threshold=79)
    span_refiner = SpanRefiner(llm=api_llm, prompts=prompts) 
    span_format = SpanFormat()
    writer = StreamWriter(output_file=output_path)
    cost_analyzer = CostAnalyze()
    
    graph = create_pipeline(annotator, correction, span_refiner, span_format, writer)

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f).get("checkpoint", 0)
    else:
        checkpoint = 0


    for idx in range(checkpoint, len(dataset)):
        item = dataset[idx]
        
        ####### CALL CUSTOM LOGIC #######
        
        if item.get('language') not in ['Italian', 'English']:
            continue
        
        # Inizializza lo stato con refined_once a False per ogni nuova esecuzione
        initial_state = {
            'text': item['text'].replace("\n", " "),
            'url': item.get('url', ''),
            'title': item.get('title', ''),
            'language': item['language'],
            'refined_once': False # Inizializza sempre a False
        }

        
        
        try:
            state = graph.invoke(initial_state)
            if state.get('error_status') is not None:
                print(f'Errore al checkpoint {idx}: {state["error_status"]}')
                break
        except Exception as e:
            print(f"Errore durante il processing: {e}")
            traceback.print_exc()  # <-- stack trace completo
            break
        
        ####### END CUSTOM LOGIC #######
        
        print(f"Salvato: {idx + 1}/{len(dataset)}")

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": idx + 1}, f, ensure_ascii=False, indent=4)

        try:
            cost_analyzer.daily_cost(threshold=llm_config['daily_cost_threshold'])
        except RuntimeError as e:
            print(f"Errore nel controllo costi: {e}")
            traceback.print_exc()  # <--  stack trace
            break