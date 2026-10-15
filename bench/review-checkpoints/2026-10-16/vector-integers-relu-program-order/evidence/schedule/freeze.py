import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
paths = ['bench/tritonbench_g/vector_addition/' + m + '.lean'
         for m in ('Proofs/TypedSchedule', 'TypedScheduleCorrect')]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def deps():
    seen, queue, external = {}, paths.copy(), set()
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        p, kind = root / rel, 'package'
        if not p.is_file():
            p, kind = repo / rel, 'repository'
        if not p.is_file():
            external.add(rel)
            continue
        assert not p.is_symlink(), 'source symlink: ' + str(p)
        seen[rel] = {'source_kind': kind, 'sha256': sha(p)}
        for line in p.read_text().splitlines():
            if line.startswith('import '):
                for imp in line[7:].split():
                    if imp.startswith('--'):
                        break
                    queue.append(imp.replace('.', '/') + '.lean')
    for rel in ('bench/tritonbench_g/vector_addition/vector_addition.py',
                'lakefile.toml', 'lake-manifest.json', 'lean-toolchain'):
        p = repo / rel
        assert p.is_file()
        seen[rel] = {'source_kind': 'repository', 'sha256': sha(p)}
    return {'files': dict(sorted(seen.items())), 'external_imports': sorted(external)}

mode = sys.argv[1]
before_path = root / 'evidence/final-dependencies-before.json'
after_path = root / 'evidence/final-dependencies-after.json'
if mode == 'before':
    assert not before_path.exists(), 'refuse frozen input overwrite'
    before_path.write_text(json.dumps(deps(), indent=2) + '\n')
    print({'modules': len(paths), 'dependency_fingerprints': len(deps()['files'])})
elif mode == 'after':
    assert not after_path.exists(), 'refuse frozen result overwrite'
    before, after = json.loads(before_path.read_text()), deps()
    assert before == after, 'source dependencies changed during validation'
    results = json.loads((root / 'evidence/verification-results.json').read_text())
    assert [r['module'] for r in results] == paths
    for r in results:
        assert r['exit_code'] == 0
        assert r['source_sha256_before'] == r['source_sha256_after'] == sha(root / r['module'])
        assert r['log_sha256'] == sha(root / r['log'])
        assert not re.search(r'\b(error|sorryAx|DISALLOWED)\b', (root / r['log']).read_text())
    after_path.write_text(json.dumps(after, indent=2) + '\n')
    print({'modules': len(results), 'all_exit_zero': True,
           'dependency_fingerprints': len(after['files']),
           'axiom_checks': sum((root / r['log']).read_text().count('axiom footprint') for r in results),
           'noncircularity_checks': sum((root / r['log']).read_text().count('non-circular ✓') for r in results)})
