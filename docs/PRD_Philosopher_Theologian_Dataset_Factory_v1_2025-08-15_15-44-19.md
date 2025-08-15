---
title: "Product Requirements Document (PRD) — Philosopher–Theologian Dataset Factory (Latin, Local, Free)"
version: "1.0"
date: "2025-08-15"
owner: "Senior Product Manager"
status: "Draft for Cross-Functional Review"
source: "Converted from BRD v1.0 (Aug 15, 2025)"
audience: ["Product", "ML/Engineering", "Data", "Research", "QA", "Legal/Compliance", "DevOps", "Community"]
license: "Docs: CC BY 4.0; Code: MIT; Datasets: CC BY-SA 4.0 (recommended)"
attribution_notice: "Include the line “Built with Meta Llama 3” in publications."
---

# 0) Context & Objectives

This PRD translates the BRD into a build-ready specification for a **local-first, zero-cost dataset factory** that generates high-quality **Latin** philosophical/theological samples for **SFT (supervised fine-tuning)** and **DPO (preference learning)**. It orchestrates **debates** among three Latin personas (Aquinas, Aristotle-in-Latin, Augustine), audits claims against a curated **retrieval stack** (BM25 + dense + optional cross-encoder), applies **quality gates**, and packages accepted turns into **provenance-rich SFT/DPO shards**. Target hardware is **Ubuntu 22.04 + RTX 4070 (12GB VRAM)**.

**Core Outcome:** A continuous, automated **dataset factory** that improves style fidelity, citation reliability, and factual grounding in Latin.

## Objectives & KPIs (from BRD → refined)

- **Support rate (auditor)** ≥ 0.80 per accepted turn
- **Latinness score** ≥ 0.20 per accepted turn (heuristic)
- **Citations/turn** in [1,2]; **misattribution** ≤ 2%
- **Novelty** Jaccard ≤ 0.85 vs cache
- **Shard validity** (JSON schema) = 100%
- **TTFB (time-to-first-batch)** from clone ≤ 30 minutes on RTX 4070
- **Reproducibility**: 100% with identical seed/config
- **Throughput**: Establish baseline; +10% in v1.1
- **Human spot-check pass** ≥ 90%

# 1) Scope, Success, Non-Goals

## In-Scope (v1.0)
- Local corpora ingestion + chunking + indexing (BM25 + FAISS dense with E5-base).
- Debate generation with strict Latin-only, citation-bound personas using local **Llama‑3‑8B‑Instruct** (fp16; 4-bit optional).
- Auditor pipeline: claim splitting → hybrid retrieval → evidence-grounded verdicts.
- Quality gates (support rate, Latinness, citation range, novelty).
- Packaging into SFT JSONL and DPO JSONL with dataset cards and full provenance.
- CLI + Makefile + `auto_runner` for end-to-end automation.
- Logs, metrics, and per-batch run summaries with deterministic packaging.

## Out-of-Scope (v1.0)
- Cloud APIs and closed models; multi-language beyond Latin; full SaaS UI; advanced human-in-the-loop UIs.

# 2) Feature Definition & Prioritization (Kano + MoSCoW)

| ID | Feature | Summary | Kano | MoSCoW | Notes |
|---|---|---|---|---|---|
| F1 | Corpora & Indexing | Ingest public-domain Latin corpora; chunk; BM25 + FAISS dense | **Basic** | **Must** | Metadata per chunk (`work`, `ref`, `text`) |
| F2 | Debate Generation | Local Llama‑3‑8B personas; YAML topics queue; word/citation bounds | **Basic** | **Must** | Latin-only, “do not invent” |
| F3 | Auditing | Claim split; hybrid retrieval; verdicts with evidence | **Performance** | **Must** | Optional cross-encoder reranker |
| F4 | Quality Gate | Thresholded acceptance by support rate, Latinness, etc. | **Performance** | **Must** | Route accepted/rejected with reasons |
| F5 | Packaging | SFT/DPO JSONL + dataset cards + provenance | **Basic** | **Must** | JSON schema + deterministic IDs |
| F6 | CLI/Automation | Makefile + run.sh + auto_runner end-to-end | **Basic** | **Must** | All configs via CLI/YAML |
| F7 | Observability & Reproducibility | Logs, metrics, batch folders, versions | **Basic** | **Must** | Seeded RNG; recorded versions |
| S1 | Cross-encoder Reranker | Optional rerank for precision | **Excitement** | **Should** | Improves auditor precision |
| S2 | LoRA Style Adapters | Persona style finetunes | **Excitement** | **Should** | Train locally; swapped at runtime |
| S3 | Regression Tests & Docker | CI-level stability + containerization | **Performance** | **Should** | Reproducible envs |
| C1 | TUI/Gradio Config UI | Friendly control surface | **Excitement** | **Could** | Non-coder UX |
| C2 | Neo4j Provenance Graph | Visualize debates/evidence | **Excitement** | **Could** | Research value |
| C3 | Metrics Dashboard | Throughput/quality in one view | **Excitement** | **Could** | Grafana/Plotly local |

# 3) Detailed Requirements by Feature

## F1. Corpora & Indexing (Must)
**Functional**
1. Load configured public-domain Latin texts (Aquinas, Augustine, Aristotle-in-Latin).
2. Chunk size **800 chars** with **80 char overlap** (configurable).
3. Build **BM25** index (e.g., `rank_bm25`) and **FAISS** dense index using **intfloat/multilingual-e5-base** embeddings.
4. Persist per-chunk metadata: `work` (title/edition), `ref` (section/paragraph anchor), `text`.
5. Record corpus and index build versions (encoder hash, faiss version).

**Non-Functional**
- Local-only; index build finishes for a 50–200 MB corpus within **≤ 30 minutes** on typical CPU.
- Idempotent rebuilds; safe to resume; checksum’d raw inputs.

**Data Contracts**
- `indices/` folder with `bm25/`, `faiss/`, `meta.json` including encoder name, dimensions, build date, checksum.

---

## F2. Debate Generation (Must)
**Functional**
1. Load **meta-llama/Meta-Llama-3-8B-Instruct** locally (fp16 default; 4-bit quant optional).
2. Read **topics queue** (`topics/queue.<batch_id>.yaml`), including:
   - `batch_id`, `seed`, `topics[]`, `turns`, `persona_order`, `min_words`, `max_words`, `min_citations`, `max_citations`.
3. For each topic, generate **persona turns** (Aquinas → Aristotle → Augustine, or configured order) of **120–180 words**, with **1–2 source citations** strictly formatted and non-invented.
4. Attach raw retrieval candidates (top-k) for transparency (optional).
5. Persist raw generations, persona prompts, and generator config under `runs/<batch_id>/generated/`.

**Non-Functional**
- Avoid OOM on 12GB VRAM: enforce `max_new_tokens`, apply 4-bit loading fallback with warning, set `torch.set_grad_enabled(False)`.
- Deterministic sampling with `seed`.

**Data Contracts**
- Generated turn JSON: `id`, `batch_id`, `topic`, `speaker`, `text`, `requested_citations_range`, `gen_config` (model, max_new_tokens, seed).

---

## F3. Auditing (Must)
**Functional**
1. **Claim splitting**: Convert each turn into a list of atomic claims (JSON array).
2. **Hybrid retrieval**: For each claim, run BM25 (k=50) and dense (k=50), merge, take top-3 per claim (configurable). Optional cross-encoder reranker.
3. **Verdicting**: For each claim, output one of:
   - `correct`, `unsupported`, `contradicted`, `misattributed`, `anachronistic`
   plus: `explanation`, `evidence_refs[]` (work/ref pairs), and `retrieval_debug` (ids/scores).
4. Save an **audit summary** per turn with `support_rate = correct / total_claims`.

**Non-Functional**
- Auditor finishes 1 turn audit in ≤ 12s on the target machine for typical claim counts.
- All steps local, no external API calls.

**Data Contracts**
- `audit_claim.json`: `{turn_id, claims:[{text, verdict, evidence_refs[]}], support_rate, notes}`

---

## F4. Quality Gate (Must)
**Functional**
- Accept a turn iff **all** thresholds pass (defaults):
  - `min_words=120`, `max_words=180`
  - `min_support_rate=0.80`
  - `min_latin_score=0.20` (stopword ratio heuristic)
  - `min_citations=1`, `max_citations=2`
  - `novelty_jaccard_max=0.85` vs cache of accepted turns
- Route to `runs/<batch_id>/accepted/` or `runs/<batch_id>/rejected/` with `gate_reason` (JSON).

**Non-Functional**
- Gate evaluation per turn ≤ 200 ms; novelty cache stored as shingles (5-grams).

**Data Contracts**
- `gate_result.json`: `{turn_id, passed: bool, metrics, thresholds, reason}`

---

## F5. Packaging (Must)
**Functional**
1. **SFT JSONL** (one item per accepted turn):  
   ```json
   {{
     "id": "<batch_id>.<uuid>",
     "instruction": "<topic or prompt>",
     "response": "<accepted turn text>",
     "meta": {{
       "speaker": "Aquinas|Aristotle|Augustine",
       "topic": "<topic>",
       "citations": [{{"work":"<title>","ref":"<section>"}}],
       "provenance": [{{"work":"<title>","ref":"<section>","snippet":"<...>"}}],
       "audit_summary": {{"claims": N, "correct": M, "support_rate": 0.83}},
       "batch_id": "<batch_id>",
       "encoder": "intfloat/multilingual-e5-base",
       "model": "Meta-Llama-3-8B-Instruct",
       "commit": "<git_sha>"
     }}
   }}
   ```
2. **DPO JSONL** (one pair per `(speaker, topic)`):  
   ```json
   {{
     "id": "<batch_id>.<uuid>",
     "prompt": "<topic or prompt>",
     "chosen": "<accepted turn text>",
     "rejected": "<nearest rejected alt or ablated version>",
     "meta": {{"audit_diffs": "...", "speaker": "...", "topic": "...", "batch_id": "..."}}
   }}
   ```
3. Generate **dataset cards** with licensing and attribution, listing all source editions used.

**Non-Functional**
- Deterministic ordering and IDs; schema validation must pass 100% before release.

**Data Contracts**
- `datasets/sft/<batch_id>.jsonl`, `datasets/dpo/<batch_id>.jsonl`, `datasets/cards/<batch_id>.md`

---

## F6. CLI & Automation (Must)
**Functional**
- `Makefile` targets: `venv`, `index`, `dense`, `debate`, `audit`, `gate`, `pack`, `all`.
- `run.sh` convenience wrapper.
- `src/auto_runner.py --config configs/default.yaml --topics topics/queue.<batch_id>.yaml` to run end-to-end.

**Non-Functional**
- Clear help messages; non-zero exit codes on failure; colorized summaries.

---

## F7. Observability & Reproducibility (Must)
**Functional**
- Persist `batch_id`, configs, versions (torch, transformers, faiss, encoder name), random seeds.
- Emit `runs/<batch_id>/summary.json` with KPIs and paths to artifacts.
- Log files per stage with timestamps; optional structured JSON logs.

**Non-Functional**
- Human-readable and machine-parseable logs; no PII; rotating log policy (configurable).

---

# 4) User Workflows & Story Mapping

## Personas
- **Home-Lab Builder (Non-coder)** → wants push-button local pipeline.
- **Scholar** → wants faithfulness in Latin style and correct citations.
- **ML Engineer/Researcher** → wants clean SFT/DPO with provenance and metrics.

## Story Map (Epics → Activities → Tasks)
- **Prepare Corpora (Epic)** → Acquire → Normalize → Chunk → Index (BM25, Dense).
- **Generate Debates (Epic)** → Configure topics YAML → Run generator → Review raw turns.
- **Audit & Gate (Epic)** → Split claims → Retrieve → Verdict → Gate.
- **Package & Release (Epic)** → Build SFT/DPO shards → Validate schemas → Create dataset cards.
- **Operate & Iterate (Epic)** → Monitor KPIs → Adjust thresholds → Re-run batches.

## Key User Journeys (Happy Paths & Friction)
1. **Home-Lab Builder: TTFB ≤ 30m**
   1) Clone repo → 2) `make venv index dense` → 3) Edit `topics/queue.<batch>.yaml` → 4) `python -m src.auto_runner ...` → 5) View `runs/<batch>/summary.json` → 6) Inspect `datasets/`.
   - **Friction risks:** CUDA/driver mismatch; VRAM OOM → **Mitigations:** 4-bit fallback, preflight checks.

2. **Scholar: Quality Assurance**
   1) Provide topic list → 2) Run `auto_runner --dry-run` to preview generated turns → 3) Inspect `audit_claim.json` and citations → 4) Approve thresholds → 5) Publish shards.
   - **Friction risks:** Citation editions differ → **Mitigations:** show `work/ref` + snippet; allow edition mapping table.

3. **ML Engineer: Training Integration**
   1) Pull SFT/DPO JSONL → 2) Validate schema with included script → 3) Train model → 4) Track gains via run IDs.
   - **Friction risks:** DPO pairs scarcity → **Mitigations:** auto-generate ablated negatives; nearest-rejected selection.

# 5) Technical Architecture

```mermaid
flowchart LR
  A[Corpora (raw Latin texts)] --> B[Chunker]
  B --> C1[BM25 Index]
  B --> C2[Dense Index (FAISS + E5)]
  D[Topics Queue YAML] --> E[Debate Generator (Llama‑3‑8B local)]
  C1 --> F[Auditor: Retrieval]
  C2 --> F
  E --> G[Claim Splitter]
  G --> F
  F --> H[Verdicts + Evidence]
  H --> I[Quality Gate]
  I -->|Accepted| J1[SFT Packer]
  I -->|Accepted| J2[DPO Packer]
  I -->|Rejected| K[Rejects + Reasons]
  J1 --> L[Datasets/SFT]
  J2 --> M[Datasets/DPO]
  subgraph Observability
    N[Logs/Run Summary/KPIs]
  end
  E --> N
  F --> N
  I --> N
```

## Components & Choices
- **BM25**: `rank_bm25` (pure-Python) for zero-cost, local search.
- **Dense**: FAISS (CPU by default, GPU optional) with E5-base embeddings via `sentence-transformers`/equivalent.
- **Reranker (opt.)**: Local cross-encoder; enable/disable by config.
- **Model Runtime**: `transformers` + `bitsandbytes` 4-bit as fallback; enforce max tokens for 12GB VRAM.
- **Data Layer**: Flat files with schema validation; deterministic IDs (batch_id + UUIDv4).
- **Reproducibility**: Persist versions and seed; `requirements.txt` and optional Docker.

## Constraints & Dependencies
- Ubuntu 22.04, Python 3.10+, CUDA drivers; GPU memory 12GB (4070).
- Public-domain Latin corpora availability and correct edition referencing.

# 6) Interfaces (CLI & Config)

## Makefile Targets
- `venv`, `index`, `dense`, `debate`, `audit`, `gate`, `pack`, `all`

## CLI Examples
```bash
# End-to-end
python -m src.auto_runner --config configs/default.yaml --topics topics/queue.<batch_id>.yaml

# Stage-wise
python -m src.chunk_and_index --config configs/default.yaml
python -m src.debate_loop --topics topics/queue.<batch_id>.yaml --seed 2025
python -m src.audit_loop --batch <batch_id>
python -m src.quality_gate --batch <batch_id>
python -m src.pack_sft --batch <batch_id>
python -m src.pack_dpo --batch <batch_id>
```

## Config YAML (Excerpt)
```yaml
batch_id: "latin_v1_001"
seed: 20250815
personas:
  order: ["Aquinas", "Aristotle", "Augustine"]
  model: "meta-llama/Meta-Llama-3-8B-Instruct"
  load:
    dtype: "fp16"
    quant_4bit: true
generator:
  min_words: 120
  max_words: 180
  min_citations: 1
  max_citations: 2
auditor:
  bm25_k: 50
  dense_k: 50
  reranker: false
gate:
  min_support_rate: 0.80
  min_latin_score: 0.20
  novelty_jaccard_max: 0.85
paths:
  corpora: "data/corpora"
  indices: "indices"
  runs: "runs"
  datasets: "datasets"
```

# 7) Data Contracts (Schemas)

## Topics Queue (YAML)
```yaml
batch_id: "latin_v1_001"
seed: 20250815
turns: 3
persona_order: ["Aquinas","Aristotle","Augustine"]
topics:
  - "De gratia et libero arbitrio"
  - "De natura boni et mali"
  - "De felicitate secundum virtutem"
bounds:
  words: {min: 120, max: 180}
  citations: {min: 1, max: 2}
```

## Audit Claim JSON (per turn)
```json
{
  "turn_id": "latin_v1_001.c5b5...",
  "claims": [
    {"text": "Gratia non tollit naturam sed perficit eam.", "verdict": "correct",
      "evidence_refs": [{"work":"Summa Theologiae I-II","ref":"q109 a2"}],
      "retrieval_debug": [{"id":"...","score":1.23}]
    }
  ],
  "support_rate": 0.83,
  "notes": "..."
}
```

## Gate Result JSON
```json
{
  "turn_id":"latin_v1_001.c5b5...",
  "passed": true,
  "metrics": {"words": 156, "latin_score": 0.27, "support_rate": 0.83, "citations": 2, "novelty": 0.62},
  "thresholds": {"min_words":120, "max_words":180, "min_support_rate":0.8, "min_latin_score":0.2, "citations":[1,2], "novelty_jaccard_max":0.85},
  "reason": "meets all thresholds"
}
```

## SFT/DPO JSONL Schemas
- **SFT item** fields: `id (str)`, `instruction (str)`, `response (str)`, `meta (object)`
- **DPO item** fields: `id (str)`, `prompt (str)`, `chosen (str)`, `rejected (str)`, `meta (object)`  
Include `speaker`, `topic`, `citations[]`, `provenance[]`, `audit_summary`, `batch_id`, `encoder`, `model`, `commit`.

# 8) Acceptance Criteria (Gherkin)

## F1 — Corpora & Indexing
```
Feature: Build indices
  Scenario: Build BM25 and FAISS indices from corpora
    Given a corpus folder with valid Latin texts and metadata
    When I run "python -m src.chunk_and_index --config configs/default.yaml"
    Then a BM25 index and a FAISS index are created under "indices/"
    And a "meta.json" records encoder name, dimensions, build date, and checksums
```

## F2 — Debate Generation
```
Feature: Generate persona turns
  Scenario: Constrained Latin turns with citations
    Given a topics YAML with words[120,180] and citations[1,2]
    When I run the debate generator with seed 20250815
    Then each generated turn is 120–180 words in Latin
    And includes 1–2 citations formatted as work/ref
    And raw outputs are saved under "runs/<batch_id>/generated/"
```

## F3 — Auditing
```
Feature: Claim-level auditing
  Scenario: Verdicts with evidence
    Given a generated turn
    When the auditor runs hybrid retrieval with bm25_k=50 and dense_k=50
    Then each claim receives a verdict in {correct, unsupported, contradicted, misattributed, anachronistic}
    And evidence_refs include work and ref for each supported claim
    And support_rate is computed and saved
```

## F4 — Quality Gate
```
Feature: Thresholded acceptance
  Scenario: Accept only high-quality turns
    Given an audited turn with support_rate >= 0.80 and latin_score >= 0.20
    And words in [120,180] and citations in [1,2]
    And novelty_jaccard <= 0.85
    When the gate runs
    Then the turn is moved to "accepted/"
    Else it is moved to "rejected/" with reason
```

## F5 — Packaging
```
Feature: Create SFT/DPO shards
  Scenario: Valid JSONL outputs
    Given a set of accepted turns
    When the packers run
    Then datasets/sft/<batch_id>.jsonl and datasets/dpo/<batch_id>.jsonl are created
    And both files pass the provided JSON schema validation
```

## F6 — CLI/Automation
```
Feature: One-command pipeline
  Scenario: End-to-end auto_runner
    Given a valid config and topics YAML
    When I run "python -m src.auto_runner --config ... --topics ..."
    Then the system completes index → debate → audit → gate → pack
    And writes runs/<batch_id>/summary.json with KPIs
```

## F7 — Observability
```
Feature: Reproducible runs
  Scenario: Deterministic with same seed
    Given identical corpora, config, and seed
    When I run the pipeline twice
    Then package IDs and shard contents are identical
```

# 9) Release Strategy & Timeline

## v1.0 (Weeks 1–8)
- M0: Prompts hardened; YAML finalized; logging and schema checks implemented.
- M1: Retrieval tuned; novelty cache; regression tests for Latinness/support-rate; Makefile end-to-end.
- M2: Minimal TUI/CLI improvements; auto dataset cards; dry-run & resume; Docker (opt.).
- M3: Human spot-check protocol; publish v1 datasets; KPI baseline captured.

## v1.1 (Weeks 9–12)
- Cross-encoder reranker enabled; LoRA style adapters (optional).
- Throughput optimization (+10%); improved dashboards; additional KPIs.

**Release Gates**
- All v1 KPIs met per batch; legal attribution verified; schema validation 100%; run summary present.

# 10) QA Strategy

- **Unit tests**: chunker, indexer, auditor verdict mapping, gate thresholds, schema validators.
- **Regression tests**: Latinness & support-rate on fixed seed topics set.
- **Smoke tests**: tiny corpus, 1-topic batch end-to-end.
- **Error taxonomy**: OOM, missing citation, low support-rate, index mismatch, schema fail.
- **Human spot-check**: 10% sample of accepted turns each batch.

# 11) Security, Compliance, Licensing

- Local-only execution; no outbound network required for pipeline runs.
- Licenses clearly recorded; include edition-level attributions in dataset cards.
- Publication must include: **“Built with Meta Llama 3”**.
- Datasets recommended under **CC BY‑SA 4.0**.

# 12) Observability & Run Summary

- **Per-batch summary (`runs/<batch_id>/summary.json`)**:  
  ```json
  {
    "batch_id":"latin_v1_001",
    "kpis":{"support_rate_avg":0.82,"latinness_avg":0.24,"citations_avg":1.6,"novelty_max":0.81},
    "counts":{"topics":6,"turns_total":18,"accepted":12,"rejected":6},
    "artifacts":{"sft":"datasets/sft/latin_v1_001.jsonl","dpo":"datasets/dpo/latin_v1_001.jsonl"},
    "versions":{"torch":"...", "transformers":"...", "faiss":"...", "encoder":"intfloat/multilingual-e5-base", "model":"Meta-Llama-3-8B-Instruct"},
    "git_sha":"<commit>"
  }
  ```

# 13) RAID Log

| Type | ID | Description | Impact | Likelihood | Mitigation | Owner | Status |
|---|---|---|---:|---:|---|---|---|
| Risk | R1 | Hallucinated citations slip past gates | High | Med | Raise support threshold; strengthen retrieval; human spot-check | QA Lead | Open |
| Risk | R2 | Retrieval recall misses chunks | Med | Med | Tune chunking/k; add cross-encoder | Tech Lead | Open |
| Risk | R3 | GPU OOM/perf on 12GB | Med | Med | 4-bit fallback; reduce max_new_tokens | Tech Lead | Open |
| Assump. | A1 | Public-domain corpora available | High | Med | Provide edition alternatives | Data Eng | Validating |
| Issue | I1 | Latinness heuristic too weak | Med | Med | Improve stopword list; add char n-gram model | Research | Investigating |
| Dep. | D1 | CUDA/torch/transformers versions | High | High | Lock reqs; provide Docker | DevOps | Tracking |

# 14) Open Questions & Decisions

- **Q1**: Which cross-encoder reranker to prefer locally? → *Decision pending v1.1 benchmarking.*
- **Q2**: Exact Latinness heuristic definition? → *Initial stopword ratio; iterate with char n-gram language ID.*
- **Q3**: DPO negatives source priority? → *Nearest rejected > ablated accepted > random.*

# 15) Traceability

- **F1** → `src/chunk_and_index.py`, `src/embeddings.py`, `indices/*`
- **F2** → `src/debate_loop.py`, `prompts/persona_*.txt`, `topics/queue.*.yaml`
- **F3** → `prompts/auditor_*.txt`, `src/audit_loop.py`
- **F4** → `src/quality_gate.py`
- **F5** → `src/pack_sft.py`, `src/pack_dpo.py`, `datasets/*`
- **F6** → `Makefile`, `run.sh`, `src/auto_runner.py`
- **F7/NFR** → `README.md`, `requirements.txt`, `runs/*`

---

## Appendix A — Defaults
`min_words=120`, `max_words=180`, `min_support_rate=0.80`, `min_latin_score=0.20`, `min_citations=1`, `max_citations=2`, `novelty_jaccard_max=0.85`

## Appendix B — Environment
Ubuntu 22.04+, Python 3.10+, CUDA-compatible GPU ~12GB VRAM.

## Appendix C — Attribution
Include “Built with Meta Llama 3” in publications; dataset cards list sources and editions.

