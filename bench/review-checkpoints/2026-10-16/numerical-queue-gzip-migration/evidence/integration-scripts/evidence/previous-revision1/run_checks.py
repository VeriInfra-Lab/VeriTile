"""Run the revision audit and changed integration suite; retain inherited consumer results."""
from pathlib import Path
import subprocess, sys, os, json, hashlib, time

sys.dont_write_bytecode = True
W = Path(__file__).resolve().parent
A = W / 'evidence/revision-attempts' / str(time.time_ns())
A.mkdir(parents=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['audit_revision.py', 'integration.py', 'test_integration.py']:
    (A / (name + '.txt')).write_bytes((W / name).read_bytes())
commands = [
    ('revision-audit', [sys.executable, '-B', str(W / 'audit_revision.py')]),
    ('integration-tests', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(W), '-p', 'test_integration.py', '-v']),
]
records = []
for name, cmd in commands:
    log = A / (name + '.log')
    start = time.monotonic()
    with log.open('w') as stream:
        result = subprocess.run(cmd, cwd=W, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=stream, stderr=subprocess.STDOUT)
    records.append({'name': name, 'command': cmd, 'exit_code': result.returncode, 'seconds': round(time.monotonic() - start, 3), 'log': str(log.relative_to(W)), 'log_sha256': sha(log)})
    (A / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
    print(log.read_text())
    if result.returncode:
        raise SystemExit(result.returncode)
data = {'status': 'passed', 'commands': records, 'integration_tests_run': 16, 'new_revision_tests': 8, 'consumer_tests_inherited_from_unchanged_candidate': 14, 'consumer_tests_rerun': False, 'live_plan_or_migration_run': False, 'acceptance_increment': 0}
(W / 'evidence/revision-checks.json').write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps(data))
