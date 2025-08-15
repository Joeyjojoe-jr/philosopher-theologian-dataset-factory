# End-to-End Run (Once Implemented)
1. Prepare corpora in `data/corpora/` (public-domain Latin).
2. Build indices:
   ```bash
   make index
   ```
3. Run debate → audit → gate → pack:
   ```bash
   python -m src.auto_runner --config configs/default.yaml --topics topics/queue.latin_v1_001.yaml
   ```
4. Validate and inspect:
   ```bash
   python -m src.validate_jsonl --input datasets/sft/latin_v1_001.jsonl --schema schemas/sft.schema.json
   python -m src.validate_jsonl --input datasets/dpo/latin_v1_001.jsonl --schema schemas/dpo.schema.json
   jq . runs/latin_v1_001/summary.json
   ```
5. Train LoRA on SFT:
   ```bash
   bash scripts/train_sft.sh
   ```
