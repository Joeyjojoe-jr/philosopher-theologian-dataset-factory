
# Business Requirements Document (BRD)
## Project: Philosopher–Theologian Dataset Factory (Latin, Local, Free)
**Version:** 1.0 • **Date:** August 15, 2025 • **Owner:** Senior Product Manager  
**Sources Reviewed:** 
- *Philosophical and Theological Data Creator concept* (attached chat log/spec)
- *philosopher_dataset_factory_*.zip (rough program concept repo with code, prompts, and Makefile)

---

## 0) Executive Summary
This BRD specifies a local-first, zero-cost pipeline that **generates high-quality Latin philosophical/theological datasets** for supervised fine-tuning (SFT) and preference learning (DPO). The system stages **debates** between three Latin personas (Aquinas, Aristotle-in-Latin, Augustine), runs a **citation-focused auditor**, applies objective **quality gates**, and packages accepted turns into **SFT/DPO shards with full provenance**. It is optimized for an Ubuntu + RTX 4070 (12GB) workstation and is fully reproducible from the command line.

**Core Outcome:** A continuous, automated **dataset factory** producing small, clean, well-provenanced Latin samples that improve model style, citation reliability, and factual grounding.

---

## 1) Stakeholder & User Analysis
### 1.1 Stakeholders
- **Product Owner / Sponsor** — Defines vision, outcomes, and budget; prioritizes roadmap.
- **Senior Product Manager (You)** — Owns BRD, backlog, KPIs; cross-functional alignment.
- **Tech Lead / ML Engineer** — Owns code design, model selection, retrieval stack, packaging.
- **Data Engineer** — Owns corpora ingestion, chunking, indexing (BM25 + FAISS), data quality.
- **Research Lead (Theology/Philosophy)** — Validates persona prompts, citation rules, corpora scope.
- **QA / Evaluation Lead** — Designs gates, audits, regression tests, human review workflows.
- **Legal & Compliance** — Reviews licenses, attribution (e.g., “Built with Meta Llama 3”), dataset cards.
- **DevOps** — Environment setup, reproducibility scripts, containerization, CI.
- **Community / OSS Liaison** — Docs, tutorials, contribution guidelines, issue triage.

### 1.2 RACI Matrix (key activities)
| Activity | Product Mgr | Tech Lead | Data Eng | Research Lead | QA Lead | Legal | DevOps | Community |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vision & Roadmap | **A** | C | I | C | I | I | I | C |
| Corpora acquisition/licensing | R | C | **A/R** | C | I | **C** | I | I |
| Indexing (BM25/dense/rerank) | C | **A** | **R** | C | I | I | I | I |
| Persona prompt design | C | C | I | **A/R** | C | I | I | I |
| Debate generation | C | **A/R** | C | C | C | I | I | I |
| Auditing (claim split + verdict) | C | **A/R** | C | **C** | C | I | I | I |
| Quality gates & scoring | C | **A** | C | C | **R** | I | I | I |
| Packaging (SFT/DPO + cards) | C | **A** | R | C | C | C | I | I |
| Release & attribution | **A** | C | I | C | I | **R** | C | C |
| CI/Env/Containerization | C | C | I | I | I | I | **A/R** | I |
| Community docs/tutorials | C | C | I | I | I | I | I | **A/R** |

**Legend:** A = Accountable, R = Responsible, C = Consulted, I = Informed

### 1.3 Primary User Personas
- **Home-Lab Builder (Non‑coder, Ubuntu + RTX 4070)**  
  *Goals:* Run locally at $0 cost; push-button pipeline; reproducible outputs.  
  *Pains:* GPU memory limits; complex Python stacks; fear of hallucinated citations.  
  *Needs:* Clear CLI/Makefile, good defaults, helpful error messages, dataset cards.

- **Theology/Philosophy Scholar**  
  *Goals:* Faithful Latin style; accurate citations; controllable topics.  
  *Pains:* Poor Latin quality; invented references.  
  *Needs:* Strict prompts, robust auditor, transparent evidence linking.

- **ML Engineer / Researcher**  
  *Goals:* Clean SFT/DPO with provenance; adjustable gates; measurable gains.  
  *Pains:* Contaminated data; irreproducible runs; opaque heuristics.  
  *Needs:* Versioned configs, logs, metrics dashboard, tests, CI.

---

## 2) Value Proposition & Differentiation
### 2.1 Value Proposition Canvas

**Customer Jobs**  
- Generate Latin theological/philosophical training pairs with **verifiable citations**.  
- Continuously grow datasets by feeding **topics queues**.  
- Package into **SFT/DPO shards** with metadata and provenance.

**Pains**  
- Hallucinated or misattributed citations.  
- Latin text scarcity and style drift.  
- Local hardware constraints (12GB VRAM).

**Gains**  
- **Automated** debate → audit → gate → package loop.  
- **Hybrid retrieval** (BM25 + dense + optional cross-encoder rerank).  
- **Quality gates** (support rate, Latinness, citations, novelty).  
- **Provenance** per turn (works/refs used for verdicts).

**Products & Services**  
- Local CLI + Makefile + scripts; persona prompts; auditor prompts.  
- Indices builder (chunking + BM25 + FAISS E5-base).  
- SFT/DPO packers; dataset cards.  
- (Optional) LoRA style adapters; cross-encoder reranker.

**Pain Relievers**  
- Auditor performs claim-splitting + evidence-based verdicts.  
- Thresholded quality gates reduce bad samples.  
- Reproducible batch IDs, YAML topics, deterministic packaging.

**Gain Creators**  
- Continuous topics ingestion; stable shards for training.  
- Full metadata for attribution and research reuse.

### 2.2 Unique Selling Points (USPs)
- **Local-first, zero-cost** pipeline targeted at RTX 4070-class GPUs.  
- **Latin-only personas** with strict, citation-bound prompts.  
- **Evidence-grounded auditing** with support-rate thresholds.  
- **Quality gating** (min_words/max_words, min_latin_score, min/max citations, novelty Jaccard).  
- **Provenance-rich SFT/DPO** output with dataset cards and licensing guidance.

---

## 3) Business Model & Market Context
### 3.1 Business Model Canvas (open-core/OSS-first)
- **Customer Segments:** Home-lab creators; theology/philosophy scholars; ML researchers; education.  
- **Value Propositions:** Verified-citation Latin datasets; automated local pipeline; reproducible artifacts.  
- **Channels:** GitHub, Hugging Face (models/datasets/spaces), academic partners.  
- **Customer Relationships:** OSS community, docs, tutorials, exemplars, Discord/Forum.  
- **Revenue Streams (optional):** Enterprise support; custom dataset services; trainings/workshops; grants/sponsorships.  
- **Key Resources:** Corpora (public domain Latin texts); Ubuntu/RTX machines; OSS libraries (Transformers/FAISS).  
- **Key Activities:** Corpus curation; indexing; persona/audit prompt tuning; QA; packaging; releases.  
- **Key Partners:** Universities, libraries, research consortia, HF community.  
- **Cost Structure:** Developer time; compute; storage; documentation/QA; CI/CD.

### 3.2 Competitive Landscape (Porter’s Five Forces)
- **Threat of New Entrants:** Medium — OSS facilitates entry, but citation rigor and Latin focus raise bar.  
- **Bargaining Power of Suppliers:** Medium — public-domain corpora availability varies by edition quality.  
- **Bargaining Power of Buyers:** Medium — alternatives exist (generic datasets), but verified Latin is niche.  
- **Threat of Substitutes:** High — generic synthetic datasets; alternative languages.  
- **Rivalry Among Competitors:** Medium — multiple dataset efforts exist, few target Latin with strict auditing.

**Risks/Barriers:** corpus quality, retrieval accuracy, legal attribution, small-GPU constraints, reproducibility.

---

## 4) Requirements & Prioritization (MoSCoW)
### 4.1 Functional Requirements
**F1. Corpora & Indexing**  
- Ingest persona corpora (Aquinas, Augustine, Aristotle-Latin).  
- Chunk texts (e.g., 800 chars w/80 overlap) and build **BM25**; build **FAISS dense** index (E5-base).  
- Store chunk metadata: `work`, `ref` (section/paragraph anchor), `text`.

**F2. Debate Generation**  
- Load local **Llama‑3‑8B‑Instruct** (fp16; 4-bit optional) with Latin persona prompts.  
- Accept YAML **topics queue** with batch_id, persona order, turns, word and citation ranges.  
- Generate persona turns (120–180 words; 1–2 citations; strict “do not invent” rules).

**F3. Auditing**  
- Split responses into **atomic claims** (JSON).  
- For each claim, perform **hybrid retrieval** (BM25 + dense; k=50 each → top-3, optional cross-encoder).  
- Produce **verdicts** per claim: `correct/unsupported/contradicted/misattributed/anachronistic` + explanation + evidence refs.

**F4. Quality Gate**  
- Gate a turn using thresholds (configurable defaults):  
  - `min_words=120`, `max_words=180`  
  - `min_support_rate=0.80` (correct/total claims)  
  - `min_latin_score=0.20` (stopword heuristic)  
  - `min_citations=1`, `max_citations=2`  
  - `novelty_jaccard_max=0.85` (anti-duplication)
- Route **accepted** vs **rejected** to separate folders with reasoning summaries.

**F5. Packaging**  
- Write **SFT JSONL**: `{{id, instruction, response, meta{{speaker,topic,citations,provenance,audit_summary}}}}`.  
- Write **DPO JSONL** pairs per `(speaker, topic)` with `prompt/chosen/rejected` + audit metadata.  
- Emit dataset cards (README-like) with licensing/attribution guidance.

**F6. CLI/Automation**  
- Provide `Makefile` targets (`venv`, `index`, `dense`, `all`) and `run.sh`.  
- Single command to process a queue end-to-end (`auto_runner.py`).  
- All paths/configs are editable via CLI flags or YAML.

**F7. Observability & Reproducibility**  
- Persist `batch_id`, logs, audits, accepted/rejected directories, and configs.  
- Deterministic packaging; record encoder/model versions.

### 4.2 Non‑Functional Requirements
- **N1. Local-first:** No external APIs; all inference/retrieval runs locally on Ubuntu 22.04+.  
- **N2. Performance:** Complete a 6-topic batch on RTX 4070 (12GB) without OOM; retry/backoff if needed.  
- **N3. Security/Compliance:** Respect licenses, include “Built with Meta Llama 3” in publications; dataset cards list sources.  
- **N4. Usability:** Copy-paste runnable commands; clear errors; documented defaults.  
- **N5. Extensibility:** Pluggable encoders/rerankers; add new personas/languages later.

### 4.3 MoSCoW Prioritization
**Must (M):** F1, F2, F3, F4, F5, F6, F7; N1–N4.  
**Should (S):** Config UI (TUI/Gradio), dataset cards automation, cross-encoder reranker, LoRA style adapters, regression tests, Docker.  
**Could (C):** Human-in-the-loop reviewer UI; Neo4j provenance graph; multilingual personas; metrics dashboard.  
**Won’t (W) for v1:** Cloud-only features; non-Latin focus; large closed models; production web SaaS.

---

## 5) Risk & Assumption Analysis
### 5.1 SWOT
- **Strengths:** Local, $0 cost; rigorous auditing; provenance; Latin niche; modular retrieval; configurable gates.  
- **Weaknesses:** Latin corpora coverage/quality; retrieval recall; small-GPU constraints; initial UX is CLI-first.  
- **Opportunities:** Education & research partnerships; expand to other languages/domains; dataset marketplace.  
- **Threats:** Competing synthetic datasets; license missteps; model regressions; perception of niche scope.

### 5.2 Risk Register (Top)
| ID | Risk | Impact | Likelihood | Mitigation | Owner | Trigger |
|---|---|---:|---:|---|---|---|
| R1 | **Hallucinated/incorrect citations** slip past gates | High | Med | Raise support-rate; strengthen retrieval; add human spot check | QA Lead | Low support-rate in audits |
| R2 | **Retrieval recall** misses relevant chunks | Med | Med | Tune chunk size/overlap; adjust k; add cross-encoder reranker | Tech Lead | Many “unsupported” verdicts |
| R3 | **GPU OOM / perf** on 12GB | Med | Med | 4-bit loading; shorter max_new; gradient-disable; batch sizing | Tech Lead | OOM errors |
| R4 | **License/attribution** errors | High | Low | Legal review; dataset cards; automated notice injection | Legal | Pre-release checklist |
| R5 | **Data drift/duplication** | Med | Med | Novelty Jaccard thresholds; shingle cache; dedupe job | Data Eng | High duplicate rate |
| R6 | **Latin quality/regressions** | Med | Med | Latinness score; regression tests; scholar review | Research | Falling Latin score |

**Assumptions:** Public-domain corpora available; Ubuntu 22.04 environment with recent CUDA; no external API reliance.

---

## 6) Success Metrics & KPIs
| Requirement | KPI | Target v1 |
|---|---|---|
| F3/F4 Auditing + Gates | **Support rate** ≥ 0.80 per accepted turn | ≥ 80% |
| F2 Persona fidelity | **Latinness score** ≥ 0.20 (heuristic) | ≥ 0.20 |
| F2/F4 Citations | Avg. **citations/turn** ∈ [1,2]; **misattribution** ≤ 2% | In-range; ≤ 2% |
| F4 Novelty | **Jaccard similarity** ≤ 0.85 vs cache | ≤ 0.85 |
| F5 Packaging | **Shard validity** (JSON schema pass) | 100% |
| F6 Usability | **Time-to-first-batch** (from clone) | ≤ 30 min |
| F7 Reproducibility | **Batch reproducibility** with same seed/config | 100% |
| N2 Performance | **Throughput** (accepted turns/hour on 4070) | Baseline +10% by v1.1 |
| QA | **Human spot-check pass rate** | ≥ 90% |

Each KPI is logged per `batch_id` and surfaced in a simple run summary (stdout + JSON).

---

## 7) Next Steps & Timeline (High Level)
**Milestone 0 – Hardening (Week 1–2)**  
- Lock prompts; parameterize thresholds; finalize YAML schema.  
- Implement logging & run summaries; add schema checks for SFT/DPO JSONL.  
- License/attribution review; draft dataset card template.

**Milestone 1 – Reliability (Week 3–4)**  
- Tune chunking (size/overlap) and k-values; optional cross-encoder rerank.  
- Add novelty cache; regression tests for Latinness/support-rate; error taxonomy.  
- Implement Makefile end-to-end (`venv → index → dense → auto_runner`).

**Milestone 2 – UX & Packaging (Week 5–6)**  
- Add simple TUI/CLI wizard or minimal Gradio controls; better error messages.  
- Auto-generate dataset cards; add `--dry-run` and `--resume`.  
- Provide Dockerfile/Compose for reproducibility.

**Milestone 3 – Evaluation & Release (Week 7–8)**  
- Human spot-check protocol; publish v1 datasets (SFT/DPO) with cards.  
- KPI baseline capture; postmortem and v1.1 roadmap (LoRA styling, dashboard).

**Dependencies/Constraints:** Public-domain corpora; CUDA drivers; torch/transformers/faiss versions; Meta Llama 3 attribution line.

---

## 8) Traceability (Requirements → Artifacts)
- **F1:** `src/chunk_and_index.py`, `src/embeddings.py`, `indices/*`  
- **F2:** `src/debate_loop.py`, `prompts/persona_*.txt`, `topics/queue.*.yaml`  
- **F3:** `prompts/auditor_*.txt`, `src/audit_loop.py`  
- **F4:** `src/quality_gate.py` (thresholds)  
- **F5:** `src/pack_sft.py`, `src/pack_dpo.py`, `datasets/*`  
- **F6:** `Makefile`, `run.sh`, `src/auto_runner.py`  
- **F7/N1–N4:** `README.md`, `requirements.txt`, logs & run folders (`runs/*`)

---

## 9) Appendices
- **A. Default Thresholds**: `min_words=120`, `max_words=180`, `min_support_rate=0.80`, `min_latin_score=0.20`, `min_citations=1`, `max_citations=2`, `novelty_jaccard_max=0.85`.  
- **B. Default Model/Encoders:** `meta-llama/Meta-Llama-3-8B-Instruct`, `intfloat/multilingual-e5-base`; (optional) cross-encoder reranker.  
- **C. Environment:** Ubuntu 22.04+, Python 3.10+, CUDA-compatible GPU ~12GB VRAM.  
- **D. Licensing Notes:** Code MIT; datasets CC BY‑SA 4.0 recommended; include “Built with Meta Llama 3” in publications; cite specific public-domain editions used.

---

**Approval:**  
- Product Owner: ☐  
- Tech Lead: ☐  
- Research Lead: ☐  
- Legal & Compliance: ☐  
- QA Lead: ☐
