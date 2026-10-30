"""Run the retained source-indexing controls against this explicit candidate."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'rms_scalar_domain_index_controls', HERE.parent / 'rms_matmul_rbe/check_source.py')
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)
controls.HERE = HERE

if __name__ == '__main__':
    controls.check()
