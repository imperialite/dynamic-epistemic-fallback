# Safer Policy Compliance with Dynamic Epistemic Fallback

Replication code and data for the paper *"Safer Policy Compliance with Dynamic Epistemic Fallback"* (Imperial & Tayyar Madabushi). [[arXiv]](https://arxiv.org/abs/2601.23094)

## Overview

Humans develop cognitive defenses known as *epistemic vigilance* to combat risks of deception and misinformation. Inspired by this mechanism, we introduce **Dynamic Epistemic Fallback (DEF)**, a dynamic safety protocol for improving an LLM's inference-time defenses against deceptive attacks that make use of maliciously perturbed policy texts. Through various levels of one-sentence textual cues, DEF nudges LLMs to flag inconsistencies, refuse compliance, and fall back to their parametric knowledge upon encountering perturbed policy texts.

This repository contains the full pipeline for:

1. **Policy perturbation**: Generating maliciously weakened versions of HIPAA and GDPR policy texts via semantic weakening (e.g., "shall" → "may optionally") and authorization weakening (e.g., "written authorization" → "verbal or informal authorization").
2. **LLM compliance evaluation**: Testing whether frontier LLMs (GPT-5-Mini, Qwen3-30B-Think, DeepSeek-R1) detect perturbed policies and produce correct compliance verdicts under various prompting strategies (zero-shot, few-shot, self-refine) with and without DEF textual cues.
3. **Chain-of-thought analysis**: Using LLM-as-judge to classify reasoning traces for detection signals and refusal behaviors.
4. **Theme categorization**: Labeling detection/refusal snippets into thematic categories (Integrity Suspicion, Logical Conflict, Textual Invalidity, Dual Resolution, Knowledge Override, Perturbed Policy Obedience).

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

