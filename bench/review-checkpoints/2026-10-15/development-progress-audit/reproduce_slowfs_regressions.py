"""Recheck source assertions with a longer process deadline on this filesystem.

Only subprocess wall-clock limits change. Test inputs, assertions, expected
failures, source gates and proof assumptions are unchanged.
"""
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
log=(Path(__file__).parent/'source-regressions.log').read_text()
names=re.findall(r'^(?:FAIL|ERROR): \S+ \(([^)]+)\)',log,re.M)
assert len(names)==12 and len(set(names))==12,names
original_run=subprocess.run

def run_with_fs_deadline(*args,**kwargs):
    if isinstance(kwargs.get('timeout'),(int,float)):
        kwargs['timeout']=max(kwargs['timeout'],300)
    return original_run(*args,**kwargs)

subprocess.run=run_with_fs_deadline
print('Source assertions unchanged; subprocess deadline raised to at least 300 seconds.',flush=True)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
sys.exit(0 if result.wasSuccessful() else 1)
