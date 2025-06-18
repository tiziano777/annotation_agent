# Annotation Agent Framework

## Overview

This repository provides a modular, extensible, flexible, and reusable AI agent framework for textual span annotation tasks, primarily targeted at Named Entity Recognition (NER) and similar applications requiring the identification of spans within a textual corpus.

The system is built on top of the **LangGraph** framework, which allows the construction and execution of graph-structured pipelines. Nodes within the pipeline can be easily modified or extended, enabling rapid adaptation to project-specific requirements.

## Architecture and Pipelines

The agent supports two primary pipeline configurations:

1. **Local Model Pipeline (via llama.cpp)**:

   * Designed for use with local inference models.
   * Suitable for privacy-sensitive projects or environments with constrained internet access.

2. **API-Based Pipelines**:

   * **With Span Refinement (Chain-of-Thought loop)**: Incorporates a feedback loop to iteratively refine span extraction.
   * **Without Span Refinement**: A faster, simpler pipeline that skips refinement steps.

These configurations enable users to choose the best production strategy based on hardware availability and privacy requirements.

## Directory Structure

```
annotation-agent/
├── config/             # Model configurations, API call setups, and prompt definitions
├── data/
│   ├── checkpoint/     # Stores checkpoint.json for resumable execution
│   ├── raw/            # Input data in JSONL format to be annotated
│   └── output/         # Annotated output in append mode
├── local_llm/          # Contiene modello Locale Llamacpp (if any) 
├── log/                # Logging of token/cost usage and execution metadata
├── nodes/              # Definition of custom LangGraph node classes
├── states/             # Graph state definitions
├── pipelines/          # Pipelines, graph definitions of flows of nodes and edges
├── utils/              # Cost estimator, helpers, and common utility scripts
├── main.py             # Execution script to run one or more pipelines
├── requirements.txt    # Required Python dependencies
└── README.md           # Project documentation
```

## Data Requirements

**Input Format**:

* The input data in `data/raw/` must be in JSON Lines (`.jsonl`) format.
* Keys and structure of these documents must be adjusted in:

  * The custom node logic files in `nodes/`
  * The `run_pipeline` function in `main.py`

**Output Format**:

The `data/output/` folder will receive annotated spans in the JSONL format (example schema)

**Checkpointing**:

* `data/checkpoint/checkpoint.json` must be initialized to enable resumability:

  ```json
  {
      "checkpoint": 0
  }
  ```
* This index tracks the item in the dataset where annotation should resume in case of interruption.

## Initialization Commands

To initialize the project correctly, run the following commands:

```bash
mkdir -p data/checkpoint
mkdir -p data/raw
mkdir -p data/output
mkdir -p logs

# Initialize the checkpoint file
cat <<EOF > data/checkpoint/checkpoint.json
{
    "checkpoint": 0
}
EOF
```

## Cost Estimation

A utility is provided in `utils/` to estimate API usage cost. Initially configured for Google's Gemini model (input/output inference), this can be easily adapted to any API service of choice.

## Customization Notes

Significant portions of the codebase are marked with `# CUSTOM LOGIC` comments. These are placeholders for adapting the agent to a specific use case. In particular, they should be revised to:

* Reflect the correct input keys for your data format
* Apply logic specific to your annotation schema

## Applications

This annotation agent has been applied to various tasks, this is a short list with my previous use cases:

* Detection of disinformation signals (My last use, code is implemeted for this task!)
* Targeted Named Entity Recognition of specific entity types
* Generic NER for the creation of pretraining corpora


## Installation

To install the required Python dependencies:

```bash
pip install -r requirements.txt
```

You can also define a .yaml fiel with settings of your API or cpp LLM, for example:

API LLM used setting:

```bash
model_name: gemini-2.0-flash-001
daily_cost_threshold: 10.0
n_ctx: 800
temperature: 0.7
top_p: 0.9
top_k: 40
max_output_tokens: 1250
api_key: YOUR_GEMINI_APIKEY
```

For Local LLM, I have used a quantized GGUF LLM model by Google Gemma 3:

```bash
model_name: "google_gemma-3-12b-it-Q8_0.gguf"
model_directory: "./llm"
#grammar_path: "./config/JSONgrammar.gbnf"
grammar_path: null
n_gpu_layers: -1
n_ctx: 2048
n_batch: 1024
verbose: false
repeat_penalty: 1.12
temperature: 0.15
top_k: 20
top_p: 0.15
streaming: false

user_input_limit: 1000
```

## Final Remarks

Feel free to use and adapt this annotation agent for any span-based annotation needs in your projects. Its flexible design allows integration with diverse language models and annotation schemes.
