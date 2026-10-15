"""Bind final verification to unchanged new, frozen ReLU and repository sources."""
from pathlib import Path
import hashlib
import json
import re
import sys

root = Path(__file__).resolve().parent
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
prior = root.parent / 'relu-original-wip'
modules = ['bench/tritonbench_g/relu_triton_kernel/' + p + '.lean' for p in
           ['Proofs/IntegerWords', 'Proofs/WordGrid', 'WordGridCorrect']]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
old = read(prior / 'evidence/receipt.json')
assert sha(prior / 'evidence/receipt.json') == 'ed4a852afafce5efd027f62565e32afe42146eb9c0b140d872990a5da211fd90'
assert sha(prior / 'evidence/scope-review.json') == 'f484d80a8fbb1ff238a4a2ebb360b5d0991938d6b79437adb9c1b6f0b8d49453'
for p,h in old['modules'].items(): assert sha(prior/p)==h
for rel in modules:
    text = (root/rel).read_text()
    assert text.endswith('\n'), 'missing final newline: ' + rel
    assert all(line == line.rstrip(' \t') for line in text.splitlines()), 'trailing whitespace: ' + rel

def dependencies():
    seen, external = {}, set()
    queue = modules + list(old['modules'])
    while queue:
        rel = queue.pop()
        if rel in seen: continue
        path, kind = root / rel, 'package'
        if not path.is_file(): path, kind = prior / rel, 'frozen_relu_package'
        if not path.is_file(): path, kind = repo / rel, 'repository'
        if not path.is_file(): external.add(rel); continue
        seen[rel] = {'source_kind':kind,'sha256':sha(path)}
        for line in path.read_text().splitlines():
            if line.startswith('import '):
                for name in line[7:].split():
                    if name.startswith('--'): break
                    queue.append(name.replace('.','/')+'.lean')
    for rel in ['bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py',
                'lakefile.toml','lake-manifest.json','lean-toolchain']:
        seen[rel] = {'source_kind':'repository','sha256':sha(repo/rel)}
    return {'files':dict(sorted(seen.items())), 'external_imports':sorted(external),
            'frozen_relu_receipt_sha256':sha(prior/'evidence/receipt.json'),
            'frozen_relu_scope_sha256':sha(prior/'evidence/scope-review.json')}

mode = sys.argv[1]
current = dependencies()
path = root / ('evidence/final-dependencies-' + mode + '.json')
assert mode in ('before','after')
path.write_text(json.dumps(current,indent=2)+'\n')
if mode == 'after':
    assert read(root/'evidence/final-dependencies-before.json') == current
    rows = read(root/'evidence/verification-results.json')
    assert [r['module'] for r in rows] == modules
    for row in rows:
        assert row['exit_code'] == 0
        assert row['source_sha256_before'] == row['source_sha256_after'] == sha(root/row['module'])
        assert row['log_sha256'] == sha(root/row['log'])
        assert not re.search(r'\b(error|sorryAx|DISALLOWED)\b|warning:', (root/row['log']).read_text())
print(json.dumps({'mode':mode,'bound_files':len(current['files']),
    'new_modules':len(modules),'frozen_modules_unchanged':len(old['modules'])}))
