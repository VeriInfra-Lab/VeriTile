"""Check the complete original body against the authoritative raw quotation.

The historical command name is retained for check_delivery.py. This command
never generates lowered expressions or rewrites Kernels.lean; invocation
metadata and proofs are reviewed separately from the pasted Python body.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from bench.jit_correspondence import check_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    results = check_manifest(ROOT, HERE / "jit-correspondence.json")
    assert len(results) == 1 and results[0]["mode"] == "pasted_body"
    print(json.dumps({"status": "PASS", "scope": "complete raw body and reviewed call metadata", "pairs": results}, indent=2))
