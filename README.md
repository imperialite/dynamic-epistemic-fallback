# Safer Policy Compliance with Dynamic Epistemic Fallback

Replication code and data for the paper *"Safer Policy Compliance with Dynamic Epistemic Fallback"* (Imperial & Tayyar Madabushi). [[arXiv]](https://arxiv.org/abs/2601.23094)

## Overview

Humans develop cognitive defenses known as *epistemic vigilance* to combat risks of deception and misinformation. Inspired by this mechanism, we introduce **Dynamic Epistemic Fallback (DEF)**, a dynamic safety protocol for improving an LLM's inference-time defenses against deceptive attacks that make use of maliciously perturbed policy texts. Through various levels of one-sentence textual cues, DEF nudges LLMs to flag inconsistencies, refuse compliance, and fall back to their parametric knowledge upon encountering perturbed policy texts.

This repository contains the full pipeline for:

1. **Policy perturbation**: Generating maliciously weakened versions of HIPAA and GDPR policy texts via semantic weakening (e.g., "shall" → "may optionally") and authorization weakening (e.g., "written authorization" → "verbal or informal authorization").
2. **LLM compliance evaluation**: Testing whether frontier LLMs (GPT-5-Mini, Qwen3-30B-Think, DeepSeek-R1) detect perturbed policies and produce correct compliance verdicts under various prompting strategies (zero-shot, few-shot, self-refine) with and without DEF textual cues.
3. **Chain-of-thought analysis**: Using LLM-as-judge to classify reasoning traces for detection signals and refusal behaviors.
4. **Theme categorization**: Labeling detection/refusal snippets into thematic categories (Integrity Suspicion, Logical Conflict, Textual Invalidity, Dual Resolution, Knowledge Override, Perturbed Policy Obedience).

## Repository Structure

```
├── code/
│   ├── script.py                  # Main experiment runner
│   ├── cot_analysis.py            # Detection/refusal rate analysis (LLM-as-judge)
│   ├── categorize_reasoning.py    # Theme labeling for reasoning snippets
│   ├── sample_for_annotation.py   # Stratified sampling for manual annotation
│   └── visualization/
│       ├── visualize_accuracy.py                    # Accuracy bar charts
│       ├── visualize_accuracy_correctonly.py         # Correct-policy-only comparison
│       ├── visualize_accuracy_correctonly_boxplot.py  # Box plots
│       ├── visualize_accuracy_correctonly_ci.py       # 95% CI plots
│       ├── visualize_themes.py                       # Per-policy theme distributions
│       ├── visualize_themes_combined.py              # Combined HIPAA+GDPR themes
│       ├── clean_rate_scores.py                      # Excel → CSV data cleaning
│       └── generate_latex_table.py                   # LaTeX table generation
├── data/
│   ├── hipaa/
│   │   ├── hipaa.csv                    # HIPAA policy sections with descriptions
│   │   ├── train_raw.csv                # Training data
│   │   ├── test_raw.csv                 # Test data
│   │   ├── test_poisoned.csv            # Perturbed test data
│   │   ├── test_correct.csv             # Correct-policy test data
│   │   ├── correct_policy/hipaa.txt     # Authentic HIPAA Privacy Rule summary
│   │   ├── authorize_weakening/         # Authorization-weakened policy texts
│   │   └── semantic_weakening/          # Semantically-weakened policy texts
│   └── gdpr/
│       ├── gdpr.csv                     # GDPR articles with descriptions
│       ├── gdpr.txt                     # Summarized GDPR articles
│       ├── train.csv                    # Training data
│       ├── test.csv                     # Test data
│       ├── test_poisoned.csv            # Perturbed test data
│       ├── test_correct.csv             # Correct-policy test data
│       ├── correct_policy/              # Authentic GDPR article texts
│       ├── authorize_weakening/         # Authorization-weakened texts
│       └── semantic_weakening/          # Semantically-weakened texts
├── output_runs/                         # Pre-computed experiment outputs (CSV)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required keys depend on which models you want to run:
- **OPENAI_API_KEY**: For GPT-5-Mini via OpenAI API
- **DEEPSEEK_API_KEY**: For DeepSeek-R1 via DeepSeek API
- **OPENROUTER_API_KEY**: For models via OpenRouter (used in analysis scripts)
- **HF_TOKEN**: For gated HuggingFace models (e.g., Qwen3-30B-Think)

Load them before running:
```bash
# Linux/Mac
export $(cat .env | xargs)

# Windows PowerShell
Get-Content .env | ForEach-Object { if ($_ -match '^([^#].+?)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
```

## Usage

### Running experiments

```bash
# Zero-shot with DEF level 1 (one-sentence epistemic vigilance cue)
python code/script.py \
    --model_id gpt-5-mini \
    --test_file data/hipaa/test_poisoned.csv \
    --policy_name data/hipaa \
    --mode zero_shot \
    --use_append_text \
    --epistemic_prompt_num 1 \
    --output_file output_runs/hipaa_gpt5mini_epis_1.csv

# Zero-shot without DEF (baseline)
python code/script.py \
    --model_id gpt-5-mini \
    --test_file data/hipaa/test_poisoned.csv \
    --policy_name data/hipaa \
    --mode zero_shot \
    --use_append_text \
    --output_file output_runs/hipaa_gpt5mini_epis_0.csv

# Few-shot with HuggingFace local model
python code/script.py \
    --model_id Qwen/Qwen3-30B-A3B \
    --test_file data/hipaa/test_poisoned.csv \
    --policy_name data/hipaa \
    --mode few_shot \
    --use_hf \
    --output_file output_runs/hipaa_qwen_fewshot.csv
```

### Analyzing reasoning traces

```bash
# Compute detection rate
python code/cot_analysis.py detection-rate output_runs/hipaa_gpt5mini_epis_1.csv --model o3

# Compute refusal rate
python code/cot_analysis.py refusal-rate output_runs/hipaa_gpt5mini_epis_1.csv --model o3

# Compute flip rate between correct and perturbed
python code/cot_analysis.py flip-rate output_runs/hipaa_gpt5mini_correct.csv output_runs/hipaa_gpt5mini_epis_0.csv

# Compute compliance distribution
python code/cot_analysis.py compliance-dist output_runs/hipaa_gpt5mini_epis_1.csv
```

### Categorizing reasoning themes

```bash
python code/categorize_reasoning.py \
    --detection-csv output_runs/hipaa_gpt5mini_epis_1_detection.csv \
    --refusal-csv output_runs/hipaa_gpt5mini_epis_1_refusal.csv \
    --out-dir results/ \
    --model gpt-5.2
```

## Models Evaluated

| Model | Identifier | Backend |
|-------|-----------|---------|
| GPT-5-Mini | `gpt-5-mini` | OpenAI API |
| Qwen3-30B-Think | `Qwen/Qwen3-30B-A3B` | HuggingFace (local) |
| DeepSeek-R1 | `deepseek-reasoner` | DeepSeek API |

## DEF Prompting Conditions

- **No DEF**: Standard zero-shot prompt (baseline)
- **DEF 1–3**: Progressively stronger one-sentence textual cues appended to the task instruction, nudging the model to flag inconsistencies, refuse compliance with suspicious policy text, and fall back to parametric knowledge when encountering perturbed policies

## Perturbation Types

- **Semantic Weakening**: Obligation modals weakened (e.g., "shall notify" → "may notify", "must ensure" → "can ensure")
- **Authorization Weakening**: Consent/authorization requirements weakened (e.g., "written authorization" → "verbal or informal authorization", "explicit consent" → "informal consent")

## Citation

```bibtex
@article{imperial2026safer,
    title={Safer Policy Compliance with Dynamic Epistemic Fallback},
    author={Imperial, Joseph Marvin and Tayyar Madabushi, Harish},
    journal={arXiv preprint arXiv:2601.23094},
    year={2026}
}
```
