import json
import subprocess
from pathlib import Path
import yaml
from src.validate_jsonl import validate_file


def test_dry_run_pipeline(tmp_path):
    cfg = yaml.safe_load(Path("configs/default.ci.yaml").read_text())
    cfg["paths"]["corpora"] = str(tmp_path / "corpora")
    cfg["paths"]["indices"] = str(tmp_path / "indices")
    cfg["paths"]["runs"] = str(tmp_path / "runs")
    cfg["paths"]["datasets"] = str(tmp_path / "datasets")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    subprocess.check_call(
        [
            "python",
            "-m",
            "src.auto_runner",
            "--config",
            str(cfg_path),
            "--topics",
            "topics/queue.latin_v1_001.yaml",
            "--dry-run",
        ]
    )

    sft_path = tmp_path / "datasets" / "sft" / "latin_v1_001.jsonl"
    assert sft_path.exists()
    first = json.loads(sft_path.read_text().splitlines()[0])
    assert first["meta"]["audit_summary"]["correct"] > 0
    assert first["meta"]["citations"]
    validate_file(str(sft_path), "schemas/sft.schema.json")
