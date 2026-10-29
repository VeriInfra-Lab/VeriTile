from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
WORK = Path('/tmp/veritile-standardization-20261015/batched-vecmat-schedule-wip')
LAKE = '/tmp/veritile-lean-toolchain/bin/lake'
LEAN = '/tmp/veritile-lean-toolchain/bin/lean'
base = subprocess.check_output([LAKE, 'env', 'printenv', 'LEAN_PATH'], cwd=ROOT, text=True).strip()
env = dict(os.environ, LEAN_PATH=str(WORK) + ':' + base)
name = sys.argv[1]
path = WORK / name
log = WORK / 'evidence' / (path.stem + '.log')
log.parent.mkdir(exist_ok=True)
command = [LEAN, '--root=' + str(WORK), str(path), '-o', str(path.with_suffix('.olean'))]
source_before = hashlib.sha256(path.read_bytes()).hexdigest()
start = time.monotonic()
with log.open('w') as stream:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
source_after = hashlib.sha256(path.read_bytes()).hexdigest()
record = {'module': name, 'source_sha256': source_after,
    'source_sha256_before': source_before, 'source_sha256_after': source_after,
    'source_unchanged_during_compile': source_before == source_after,
    'command': command, 'LEAN_PATH': env['LEAN_PATH'], 'exit_code': result.returncode,
    'seconds': round(time.monotonic() - start, 3), 'log': log.relative_to(WORK).as_posix(),
    'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()}
(log.with_suffix('.json')).write_text(json.dumps(record, indent=2) + '\n')
print(log.read_text())
print(json.dumps(record, indent=2))
sys.exit(result.returncode if source_before == source_after else 2)
