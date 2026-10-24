import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent
producer = root.parent / 'vector-addition-dtype-extension-wip'
relative = 'bench/tritonbench_g/vector_addition/IntegerGridCorrect.lean'
source = root / relative
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = json.loads((producer / 'evidence/receipt.json').read_text())
for p,h in receipt['modules'].items():
    assert sha(producer / p) == h
env = dict(os.environ)
env['LEAN_PATH'] = str(producer) + ':' + env.get('LEAN_PATH', '')
before = sha(source)
command = ['/tmp/veritile-lean-toolchain/bin/lean', '--root=' + str(root),
           str(source), '-o', str(source.with_suffix('.olean'))]
log = root / 'compile.log'
start = time.monotonic()
with log.open('w') as out:
    result = subprocess.run(command, env=env, stdout=out, stderr=subprocess.STDOUT)
for p,h in receipt['modules'].items():
    assert sha(producer / p) == h
record = {'module': relative, 'source_sha256_before': before, 'source_sha256_after': sha(source),
    'command': command, 'exit_code': result.returncode, 'seconds': time.monotonic() - start,
    'log': str(log), 'log_sha256': sha(log), 'dependency_source_sha256': receipt['modules'],
    'dependency_receipt_sha256': sha(producer / 'evidence/receipt.json')}
(root / 'compile-receipt.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
print(log.read_text())
raise SystemExit(result.returncode)
