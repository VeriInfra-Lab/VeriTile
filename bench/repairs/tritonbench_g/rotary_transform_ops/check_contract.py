"""Reuse the actual-source transform CPU storage interpreter."""
from pathlib import Path
import importlib.util
path=Path(__file__).resolve().parent.parent/'rotary_transform/check_contract.py'
spec=importlib.util.spec_from_file_location('transform_checks',path)
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
if __name__=='__main__':checks.check_task('rotary_transform_ops')
