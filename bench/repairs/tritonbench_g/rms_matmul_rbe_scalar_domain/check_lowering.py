"""Compile the candidate's unchanged actual JIT AST across both storage routes."""
from pathlib import Path
import argparse
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'rms_scalar_domain_native_controls', HERE.parent / 'rms_matmul_rbe/check_lowering.py')
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    controls.check(HERE / 'rms_matmul_rbe_repaired.py', args.output_dir.resolve())
