"""Exercise this task's actual host/JIT AST with the shared CPU harness."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent.parent / "flash_decode2_phi" / "check_contract.py"))["checks"]("flash_decode2_llama")
