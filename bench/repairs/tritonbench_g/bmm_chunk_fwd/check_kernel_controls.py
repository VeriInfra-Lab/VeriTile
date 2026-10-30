"""Replay the accepted wrapper controls against the single integer-index candidate."""
import importlib.util
from pathlib import Path
from check_kernel_lowering import check_candidate

here = Path(__file__).resolve().parent
source, candidate = check_candidate()
spec = importlib.util.spec_from_file_location('bmm_fwd_controls',here/'check_contract.py')
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)
controls.source_binding()
controls.SOURCE = candidate
controls.source_binding = check_candidate
controls.main()
print('Candidate replay passed; the only source edit is integer multiplication commutation in pid_h.')
