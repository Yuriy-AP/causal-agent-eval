# CausalAgentBench

Portfolio project: a financial analyst AI agent with structured evaluation using Phoenix tracing and LLM-as-judge scoring.

## Motivation

Evaluating AI agents is fundamentally a causal inference problem. Standard metrics (accuracy, F1, LLM-as-judge scores) tell you *what* happened but not *why* — and they can't isolate the incremental impact of specific changes (prompt version, temperature, model choice etc.) from confounding factors (such as input query complexity, token length, provider latency, time-of-day API variability, and the non-deterministic nature of LLM outputs themselves). 

This project applies causal inference methods to agent evaluation: treating agent configuration changes as treatments, logging structured observational data via Phoenix tracing, and building toward a measurement framework that distinguishes true performance improvements from noise.

**Current:** Agent infrastructure, Phoenix tracing, LLM-as-judge scoring.  
**Next:** Causal analysis — difference-in-differences and double ML to attribute performance changes to specific interventions.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_key_here
```

## Structure

- `agents/` — Financial analyst agent with tool-calling
- `tracking/` — LLM-as-judge evaluation logger
- `notebooks/` — Evaluation demo and analysis
- `experiments/` — Run artifacts
- `analysis/` — Causal inference outputs (future)

## Quick start

```bash
jupyter notebook notebooks/01_agent_evaluation_demo.ipynb
```
