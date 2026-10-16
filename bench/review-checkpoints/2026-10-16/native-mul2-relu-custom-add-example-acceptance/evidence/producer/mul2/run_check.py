import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib, json, subprocess, time
W = Path(__file__).resolve().parent
P = W / sys.argv[1]
A = W / 'evidence/script-attempts'
A.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
stem = str(len(list(A.glob('*.json')))+1).zfill(4)+'-'+P.stem
snapshot = A/(stem+'.py.txt');snapshot.write_bytes(P.read_bytes())
log = A/(stem+'.log')
cmd = [sys.executable, '-B', str(P), *sys.argv[2:]]
start = time.monotonic()
with log.open('w') as out:
    r = subprocess.run(cmd, cwd=W, stdout=out, stderr=subprocess.STDOUT)
record = {'script': str(P.relative_to(W)), 'source_snapshot': str(snapshot.relative_to(W)),
          'source_sha256': sha(snapshot), 'source_unchanged': sha(P)==sha(snapshot),
          'command': cmd, 'exit_code': r.returncode, 'seconds': time.monotonic()-start,
          'log': str(log.relative_to(W)), 'log_sha256': sha(log)}
(A/(stem+'.json')).write_text(json.dumps(record, indent=2)+'\n')
print(log.read_text());print(json.dumps(record));raise SystemExit(r.returncode)
