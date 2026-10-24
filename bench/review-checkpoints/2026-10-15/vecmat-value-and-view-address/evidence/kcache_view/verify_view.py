import hashlib, json, os, pathlib, subprocess, sys, time
root = pathlib.Path('/tmp/veritile-standardization-20261015/kcache-view-wip')
repo = pathlib.Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
modules = [
    'bench/tritonbench_g/kcache_copy_triton/Proofs/HostViewShape.lean',
    'bench/tritonbench_g/kcache_copy_triton/Proofs/HostViewCases.lean',
    'bench/tritonbench_g/kcache_copy_triton/Proofs/HostView.lean',
]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
records = []
for module in modules:
    source = root / module
    before = sha(source)
    log = root / 'evidence' / (source.stem + '.compile.log')
    command = [
        '/tmp/veritile-lean-toolchain/bin/lake', 'env', 'bash', '-c',
        'export LEAN_PATH="$1:$LEAN_PATH"; exec "$2" --root="$1" "$1/$3" -o "$1/${3%.lean}.olean"',
        'view-verification', str(root), '/tmp/veritile-lean-toolchain/bin/lean', module,
    ]
    start = time.monotonic()
    with log.open('w') as output:
        completed = subprocess.run(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
    record = {
        'module': module, 'exit_code': completed.returncode,
        'source_sha256_before': before, 'source_sha256_after': sha(source),
        'log': str(log.relative_to(root)), 'log_sha256': sha(log),
        'elapsed_seconds': round(time.monotonic()-start, 3), 'command': command,
    }
    records.append(record)
    (root / 'evidence' / 'verification-results.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['module','exit_code','elapsed_seconds']}), flush=True)
    if completed.returncode != 0 or before != sha(source):
        sys.exit(1)
