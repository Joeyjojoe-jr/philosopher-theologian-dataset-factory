# Philosopher–Theologian Dataset Factory (Latin, Local, Free)
**Repo Version:** v1 • **Scaffold Timestamp:** 20250815_154500

> Local-first pipeline to generate high-quality **Latin** philosophical/theological datasets via
> three-persona debates → auditing → quality gates → SFT/DPO packaging.

## Quickstart (Dry Run: no GPU needed)
```bash
# 1) Create and activate venv
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 2) Install minimal deps for dry-run and schema validation
pip install -r requirements-min.txt

# 3) Run dry-run smoke test (creates example artifacts without ML)
python -m src.auto_runner --config configs/default.yaml --topics topics/queue.latin_v1_001.yaml --dry-run

# 4) Inspect outputs
ls -R runs/latin_v1_001
python -m src.validate_jsonl --input datasets/sft/latin_v1_001.jsonl --schema schemas/sft.schema.json
python -m src.validate_jsonl --input datasets/dpo/latin_v1_001.jsonl --schema schemas/dpo.schema.json
```

## Full Pipeline (GPU, local-only) — to be completed by AI/you
```bash
# Recommended (installs FAISS, transformers, etc.)
pip install -r requirements.txt
# Build indices and run end-to-end
make index dense
python -m src.auto_runner --config configs/default.yaml --topics topics/queue.latin_v1_001.yaml
```

See `DOCS.md` for architecture and stage descriptions. This repo is aligned to your BRD/PRD.


## Push to GitHub (Versioning)
```bash
# Create a new GitHub repo (via web UI), then:
git init
git branch -m main
git add .
git commit -m "feat: initial scaffold with dry-run + schemas + CI"
git remote add origin https://github.com/<you>/philosopher-theologian-dataset-factory.git
git push -u origin main
```