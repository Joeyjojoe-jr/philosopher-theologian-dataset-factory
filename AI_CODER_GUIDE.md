# AI Coder Guide (Codex/Gemini/Claude/DeepSeek)
**Goal:** Complete the local-first Latin dataset factory so it generates real (non-dry-run) SFT/DPO from three-persona debates with audited citations.

## Constraints
- Local-first, offline; models must run on consumer GPU (12GB VRAM) with 4-bit fallback.
- All outputs must pass `schemas/*.json` via `src/validate_jsonl.py`.
- Acceptance criteria live in PRD/BRD (see docs/) and ROADMAP.md.

## Required Tasks (Implement in this order)
1. **F1 – chunk_and_index**
   - Implement `src/chunk_and_index.py` to build BM25 (rank_bm25) and FAISS over E5 embeddings.
   - Write `indices/meta.json` with encoder name, dim, faiss index type, doc count, and checksums.
2. **F2 – debate_loop**
   - Load `Meta-Llama-3-8B-Instruct` via `transformers` with 4-bit load if VRAM < 16GB.
   - Enforce Latin-only, 120–180 words, 1–2 real citations (from indices). Save per-turn JSON files.
3. **F3 – audit_loop**
   - Split claims, run hybrid retrieval, produce verdicts per claim with support_rate.
4. **F4 – quality_gate**
   - Enforce thresholds from `configs/default.yaml` and route accepted/rejected accordingly.
5. **F5 – pack_sft / pack_dpo**
   - Read accepted/rejected and write JSONL shards passing schemas, with deterministic IDs.
6. **F6 – auto_runner**
   - Wire the above, add `--resume`, nonzero exit on failures, and richer `runs/<batch>/summary.json`.
7. **F7 – training (SFT/LoRA)**
   - Implement `training/train_sft_lora.py` (TRL/PEFT) to fine-tune on SFT shards; write `models/<tag>/`.

## Coding Standards
- One module per feature; keep pure functions testable.
- Log versions & seeds into `runs/<batch>/summary.json`.
- Add unit tests where helpful; keep CI green.

## Done = Acceptance
- CI passes on dry-run and basic real-run on a tiny sample corpus.
- Shards validate and quality KPIs meet thresholds.
