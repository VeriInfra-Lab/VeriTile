"""Execute the single-projection source and wrapper with the shared CPU adapter."""
import importlib.util
from pathlib import Path
import warnings

if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "rms_matmul_rbe/check_source.py"
    spec = importlib.util.spec_from_file_location("rms_source_checks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mod.check(single=True)
