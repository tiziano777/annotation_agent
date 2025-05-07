import json
import os

from langgraph.graph import StateGraph, START, END
from states.state import State

from nodes.apiAnnotator import Annotator
from nodes.OutputCorrection import OutputCorrection
from nodes.SpanFormat import SpanFormat
from nodes.StreamWriter import StreamWriter

from utils.GeminiCostAnalyze import GeminiCostAnalyze

def create_pipeline(annotator, output_correction, span_format, writer):
    workflow = StateGraph(State)
    workflow.add_node("annotator_node", annotator)
    workflow.add_node("correction_node", output_correction)
    workflow.add_node("span_node", span_format)
    workflow.add_node("writer_node", writer)
    workflow.add_edge(START, "annotator_node")
    workflow.add_edge("annotator_node", "correction_node")
    workflow.add_edge("correction_node", "span_node")
    workflow.add_edge("span_node", "writer_node")
    workflow.add_edge("writer_node", END)
    return workflow.compile()


def run_pipeline(input_path, output_path, checkpoint_path, api_llm, prompts, llm_config):
    annotator = Annotator(llm=api_llm, prompts=prompts, input_context=llm_config['n_ctx'])
    correction = OutputCorrection(similarity_threshold=79)
    span_format = SpanFormat()
    writer = StreamWriter(output_file=output_path)
    cost_analyzer = GeminiCostAnalyze()
    graph = create_pipeline(annotator, correction, span_format, writer)

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f).get("checkpoint", 0)
    else:
        checkpoint = 0

    for idx in range(checkpoint, len(dataset)):
        item = dataset[idx]
        if item.get('language') not in ['Italian', 'English']:
            continue

        try:
            state = graph.invoke({
                'text': item['text'].replace("\n", " "),
                'url': item.get('url', ''),
                'title': item.get('title', ''),
                'language': item['language']
            })
            if state.get('error_status') is not None:
                print(f'Errore al checkpoint {idx}: {state["error_status"]}')
                break
        except Exception as e:
            print(f"Errore durante il processing: {e}")
            break

        print(f"Salvato: {idx + 1}/{len(dataset)}")

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"checkpoint": idx + 1}, f, ensure_ascii=False, indent=4)

        try:
            cost_analyzer.daily_cost(threshold=llm_config['daily_cost_threshold'])
        except RuntimeError as e:
            print(f"Errore nel controllo costi: {e}")
            break
