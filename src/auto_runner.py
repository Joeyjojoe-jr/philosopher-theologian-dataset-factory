"""
File: src/auto_runner.py
Purpose: End-to-end runner. In --dry-run, synthesizes artifacts conforming to schemas.
CLI:
  python -m src.auto_runner --config configs/default.yaml --topics topics/queue.latin_v1_001.yaml [--dry-run]
"""
import argparse, json, uuid, os, random
from pathlib import Path
from .config import load_config
from .utils.logging import write_json, now_iso

ENCODER_NAME = "intfloat/multilingual-e5-base"

def _load_topics(path: str):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--dry-run", action="store_true", help="Generate placeholder artifacts without ML")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["personas"]["model"].split('/')[-1]
    topics = _load_topics(args.topics)
    batch_id = cfg["batch_id"]

    runs_dir = Path(cfg["paths"]["runs"]) / batch_id
    datasets_dir = Path(cfg["paths"]["datasets"])

    # Ensure dirs
    (runs_dir / "generated").mkdir(parents=True, exist_ok=True)
    (runs_dir / "audits").mkdir(parents=True, exist_ok=True)
    (runs_dir / "accepted").mkdir(parents=True, exist_ok=True)
    (runs_dir / "rejected").mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    (datasets_dir / "sft").mkdir(parents=True, exist_ok=True)
    (datasets_dir / "dpo").mkdir(parents=True, exist_ok=True)
    (datasets_dir / "cards").mkdir(parents=True, exist_ok=True)

    # DRY RUN: fabricate 3 topics × 3 speakers = 9 turns
    words = (cfg["generator"]["min_words"] + cfg["generator"]["max_words"]) // 2
    speakers = cfg["personas"]["order"]
    sft_items = []
    dpo_items = []

    for t in topics["topics"]:
        accepted_text = f"({words} verba Latine ficta) {t}. Citationes verae in versione plenaria addentur."
        rejected_text = f"Textus reiectus pro DPO ad {t} (exempli gratia)."
        for sp in speakers:
            turn_id = f"{batch_id}.{uuid.uuid4().hex[:8]}"
            audit = {"claims": 5, "correct": 4, "support_rate": 0.8}
            sft = {
                "id": turn_id,
                "instruction": t,
                "response": accepted_text,
                "meta": {
                    "speaker": sp,
                    "topic": t,
                    "citations": [{"work":"Summa Theologiae I-II","ref":"q109 a2"}],
                    "provenance": [{"work":"Summa Theologiae I-II","ref":"q109 a2","snippet":"gratia non tollit naturam"}],
                    "audit_summary": audit,
                    "batch_id": batch_id,
                    "encoder": ENCODER_NAME,
                    "model": model_name,
                    "commit": ""
                }
            }
            sft_items.append(sft)
            dpo_items.append({
                "id": f"{batch_id}.{uuid.uuid4().hex[:8]}",
                "prompt": t,
                "chosen": accepted_text,
                "rejected": rejected_text,
                "meta": {"speaker": sp, "topic": t, "batch_id": batch_id, "audit_diffs": "stub"}
            })

    # Write shard files
    sft_path = datasets_dir / "sft" / f"{batch_id}.jsonl"
    dpo_path = datasets_dir / "dpo" / f"{batch_id}.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for it in sft_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(dpo_path, "w", encoding="utf-8") as f:
        for it in dpo_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Write summary
    summary = {
        "batch_id": batch_id,
        "kpis":{"support_rate_avg":0.8,"latinness_avg":0.25,"citations_avg":1.5,"novelty_max":0.8},
        "counts":{"topics":len(topics["topics"]),"turns_total":len(topics["topics"])*len(speakers),"accepted":len(topics["topics"])*len(speakers),"rejected":0},
        "artifacts":{"sft":str(sft_path),"dpo":str(dpo_path)},
        "versions":{"encoder":ENCODER_NAME,"model":model_name},
        "created_at": now_iso()
    }
    write_json(runs_dir/"summary.json", summary)
    print(f"[dry-run] Wrote {sft_path} and {dpo_path}\nSummary: {runs_dir/'summary.json'}")

if __name__ == "__main__":
    main()
