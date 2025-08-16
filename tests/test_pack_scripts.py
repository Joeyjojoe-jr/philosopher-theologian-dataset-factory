import json
import subprocess
import sys
from pathlib import Path


def test_pack_scripts_custom_paths(tmp_path):
    runs_dir = tmp_path / "runs"
    datasets_dir = tmp_path / "datasets"
    acc_dir = runs_dir / "batch1" / "accepted"
    rej_dir = runs_dir / "batch1" / "rejected"
    acc_dir.mkdir(parents=True)
    rej_dir.mkdir(parents=True)

    accepted = [
        {
            "instruction": "t1",
            "response": "r1",
            "meta": {"speaker": "s", "topic": "t1"},
        },
        {
            "instruction": "t2",
            "response": "r2",
            "meta": {"speaker": "s", "topic": "t2"},
        },
    ]
    for i, data in enumerate(accepted):
        (acc_dir / f"{i}.json").write_text(json.dumps(data), encoding="utf-8")

    rejected = {
        "instruction": "t1",
        "response": "bad",
        "meta": {"speaker": "s", "topic": "t1"},
    }
    (rej_dir / "0.json").write_text(json.dumps(rejected), encoding="utf-8")

    result_sft = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.pack_sft",
            "--batch",
            "batch1",
            "--runs-dir",
            str(runs_dir),
            "--datasets-dir",
            str(datasets_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result_sft.returncode == 0, result_sft.stdout + result_sft.stderr
    sft_path = datasets_dir / "sft" / "batch1.jsonl"
    assert sft_path.exists()
    assert len(sft_path.read_text(encoding="utf-8").strip().splitlines()) == 2

    result_dpo = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.pack_dpo",
            "--batch",
            "batch1",
            "--runs-dir",
            str(runs_dir),
            "--datasets-dir",
            str(datasets_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result_dpo.returncode == 0, result_dpo.stdout + result_dpo.stderr
    dpo_path = datasets_dir / "dpo" / "batch1.jsonl"
    assert dpo_path.exists()
    assert len(dpo_path.read_text(encoding="utf-8").strip().splitlines()) == 2
