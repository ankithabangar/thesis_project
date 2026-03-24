# Improving Meeting Summarization via LLM Self-Reflection

A thesis project exploring how iterative self-reflection can improve the factual consistency and quality of abstractive meeting summaries.

## Overview

Long meeting transcripts are difficult to summarize accurately — LLMs often introduce factual inconsistencies or miss key points. This project investigates whether a **critic-refiner self-reflection loop** can detect and fix such errors to produce better summaries.

The pipeline:
1. Preprocesses meeting transcripts from the [QMSum dataset](https://github.com/Yale-LILY/QMSum)
2. Generates baseline chunk-based summaries using **Llama 3.1 (8B)**
3. Applies a self-reflection loop where the LLM acts as its own critic and refiner
4. Evaluates quality using four metrics: ROUGE, BERTScore, SummaC, and MiniCheck


## Project Structure

```
thesis_project/
├── data/
│   └── preprocessed/
│       └── cleaned_meetings.json   # 35 cleaned & chunked test meetings
├── results/llama3.1/
│   ├── baseline_summaries.json           # Baseline generated summaries
│   ├── baseline_evaluation_results.json  # Baseline metric scores
│   └── self_refined/
|       └── iteration_1/
│           └── self_refined_summaries.json          # Refined summaries after 1 loop
│   └── evaluation_results/iteration_3/
|       └── iteration_1/
│           └── comparison_results.json    # Comparison: baseline vs refined
├── preprocess.py        # Clean transcripts, chunk at ~1000 words
├── baseline.py          # Generate initial summaries per chunk, then merge
├── self_reflection.py   # Critic-refiner loop for iterative refinement
├── evaluate.py          # Evaluate baseline summaries
└── evaluate_comparison.py  # Compare baseline vs refined summaries
```

## Pipeline

```
Raw QMSum Data
     │
     ▼
preprocess.py  →  cleaned_meetings.json (35 meetings, chunked)
     │
     ▼
baseline.py  →  baseline_summaries.json
     │
     ├──→  evaluate.py  →  baseline_evaluation_results.json
     │
     ▼
self_reflection.py  →  self_refined_summaries.json
     │
     ▼
evaluate_comparison.py  →  comparison_results.json (baseline vs refined deltas)
```

## Dataset

**QMSum** — a query-based meeting summarization dataset covering three domains:

| Domain     | Meetings | Description                      |
|------------|----------|----------------------------------|
| Academic   | 9        | Graduate student research discussions |
| Product    | 20       | Product design meetings          |
| Committee  | 6        | Government/policy committee meetings |

- Meetings range from **2,293 to 23,826 words**
- Transcripts are chunked into ~1,000 word segments (avg. 11.2 chunks/meeting)
- Each meeting has a human-written **ground truth summary** used for evaluation

## Method

### Baseline Summarization
Each transcript chunk is summarized independently (max 30 words/chunk), then chunk summaries are merged into a final meeting summary (max 150 words) using Llama 3.1.

### Self-Reflection Loop
1. **Critic**: The LLM reads the original transcript chunk and its summary, then identifies any factual inconsistencies or missing information.
2. **Refiner**: If issues are found, the LLM rewrites the summary to fix them.
3. The loop runs for a maximum of 3 iterations (early stopping if no issues are detected).

## Evaluation Metrics

| Metric       | Type             | Description                                     |
|--------------|------------------|-------------------------------------------------|
| ROUGE-1/2/L  | Reference-based  | N-gram overlap with ground truth summary        |
| BERTScore    | Reference-based  | Semantic similarity (precision, recall, F1)     |
| SummaC       | Source-based     | Factual consistency via NLI decomposition       |
| MiniCheck    | Source-based     | Claim verification using RoBERTa-large          |


## Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) with Llama 3.1 pulled locally

```bash
ollama pull llama3.1:8b
```

### Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install langchain langchain-ollama rouge-score bert-score summac minicheck nltk
```

### Run the Pipeline

```bash
# Step 1: Preprocess
python preprocess.py

# Step 2: Generate baseline summaries
python baseline.py

# Step 3: Evaluate baseline
python evaluate.py

# Step 4: Apply self-reflection
python self_reflection.py

# Step 5: Evaluate and compare
python evaluate_comparison.py
```

## Model

All summarization and reflection steps use **Llama 3.1 (8B)** running locally via Ollama. No external API calls are required.
