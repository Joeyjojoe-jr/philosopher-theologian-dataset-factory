# Documentation

## Purpose
This repository is a **build-ready scaffold** for the Philosopher–Theologian Dataset Factory.
It encodes interfaces, file contracts, and CLI entrypoints for an AI or developer
to finish the implementation quickly.

## Stages
1. **Corpora & Indexing** (`src/chunk_and_index.py`, `indices/`)
2. **Debate Generation** (`src/debate_loop.py`, `prompts/persona_*.txt`)
3. **Auditing** (`src/audit_loop.py`, `prompts/auditor_*.txt`)
4. **Quality Gate** (`src/quality_gate.py`)
5. **Packaging** (`src/pack_sft.py`, `src/pack_dpo.py`, `datasets/`)
6. **Automation** (`src/auto_runner.py`, `Makefile`, `run.sh`)
7. **Observability** (`runs/<batch_id>`, logs, versions, `summary.json`)

## Dry-Run Philosophy
The **dry run** produces valid artifacts matching the schemas without running models.
This lets you validate the pipeline end-to-end on any machine, then enable real ML later.

## Licensing
- Code: MIT (see `LICENSE`)
- Docs: CC BY 4.0
- Datasets produced: recommend CC BY‑SA 4.0; include edition attributions and “Built with Meta Llama 3”.
