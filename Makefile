# Makefile — convenience targets
.PHONY: venv index dense debate audit gate pack all smoke

venv:
	python3 -m venv .venv && . .venv/bin/activate && python -m pip install --upgrade pip

index:
	python -m src.chunk_and_index --config configs/default.yaml

dense:
	# placeholder for dense index specifics if separate

debate:
	python -m src.debate_loop --topics topics/queue.latin_v1_001.yaml

audit:
	python -m src.audit_loop --batch latin_v1_001

gate:
	python -m src.quality_gate --batch latin_v1_001

pack:
	python -m src.pack_sft --batch latin_v1_001 && python -m src.pack_dpo --batch latin_v1_001

all: index dense debate audit gate pack

smoke:
	python -m src.auto_runner --config configs/default.yaml --topics topics/queue.latin_v1_001.yaml --dry-run
