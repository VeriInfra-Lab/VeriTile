import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
env = dict(os.environ)
env['LEAN_PATH'] = str(root) + ':/tmp/veritile-standardization-20261015/vector-addition-dtype-extension-wip:/tmp/veritile-standardization-20261015/vector-addition-wip:' + env.get('LEAN_PATH', '')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
results = []
for relative in sys.argv[1:]:
    source = root / relative
    log = root / 'evidence' / (source.stem + '.compile.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    before = sha(source)
    command = ['/tmp/veritile-lean-toolchain/bin/lean', '--root=' + str(root),
               str(source), '-o', str(source.with_suffix('.olean'))]
    start = time.monotonic()
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
    row = {'module': relative, 'command': command, 'exit_code': result.returncode,
           'seconds': time.monotonic() - start,
           'source_sha256_before': before, 'source_sha256_after': sha(source),
           'log': str(log.relative_to(root)), 'log_sha256': sha(log)}
    results.append(row)
    (root / 'evidence' / 'verification-results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(row), flush=True)
    print(log.read_text(), flush=True)
    if before != sha(source):
        raise RuntimeError('source changed during compilation')
    if result.returncode:
        raise SystemExit(result.returncode)
