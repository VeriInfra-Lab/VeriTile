"""Independent, read-only integrity review of the frozen ReLU source package."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
TMP = Path('/tmp/veritile-standardization-20261015')
PKG = TMP / 'relu-source-coupled-block-wip'
OUT = TMP / 'relu-source-coupled-block-independent-review.json'
sys.path.insert(0, str(ROOT / 'bench'))
from adapter_common import declarations

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def check(p, h):
    p = Path(p)
    assert p.is_file() and sha(p) == h, str(p)

assert not OUT.exists(), 'refuse to overwrite independent review'
check(PKG / 'evidence/receipt.json', 'ad2051d04165c629830a901f25ccd73652254b7453dd3d04f183861b74856fc7')
check(PKG / 'evidence/scope-review.json', 'e647c106a1cf7fc3f3e95b8331ee573d344b9359ca80c2ba019c100c60291e5c')
receipt = read(PKG / 'evidence/receipt.json')
scope = read(PKG / 'evidence/scope-review.json')
assert receipt['whole_original_program_accepted'] is False
assert receipt['original_program_completion_increment'] == 0
for rel, h in receipt['artifacts'].items():
    check(PKG / rel, h)
assert {p.relative_to(PKG).as_posix() for p in PKG.rglob('*')
        if p.is_file() and not p.is_symlink() and p.suffix not in {'.olean', '.pyc'}
        and p != PKG / 'evidence/receipt.json'} == set(receipt['artifacts'])
for rel, h in receipt['modules'].items():
    check(PKG / rel, h)
    source = (PKG / rel).read_text()
    assert source.endswith('\n') and not re.search(r'[ \t]+$', source, re.M)
assert len(receipt['modules']) == 6
assert receipt['public_contracts'] == scope['public_contracts']
for c in receipt['public_contracts']:
    actual = declarations(PKG / c['file'])[c['parser_name']]
    assert actual['kind'] == c['kind'] == 'specification'
    assert actual['statement_sha256'] == c['statement_sha256']
    assert len(c['profiles']) == 9 and len(c['unclosed_numeric_callbacks']) == 3

before = read(PKG / 'evidence/final-dependencies-before.json')
after = read(PKG / 'evidence/final-dependencies-after.json')
assert before == after
inputs = {}
olean_count = 0
for rel, entry in after['files'].items():
    check(entry['source_path'], entry['sha256'])
    directory = PKG if entry['source_kind'] == 'package' else (
        ROOT if entry['source_kind'] == 'repository' else TMP / entry['source_kind'])
    assert directory / rel == Path(entry['source_path'])
    if entry['source_kind'] != 'package':
        check(ROOT / rel, entry['sha256'])
    inputs[rel] = entry['sha256']
    if 'olean_path' in entry:
        check(entry['olean_path'], entry['olean_sha256'])
        olean_count += 1
assert len(inputs) == 190 and olean_count == 179
seen, external, queue = set(), set(), list(receipt['modules'])
while queue:
    rel = queue.pop()
    if rel in seen or rel in external:
        continue
    entry = after['files'].get(rel)
    if entry is None:
        external.add(rel)
        assert not (ROOT / rel).exists()
        continue
    seen.add(rel)
    for line in Path(entry['source_path']).read_text().splitlines():
        if line.startswith('import '):
            for name in line[7:].split():
                if name.startswith('--'):
                    break
                queue.append(name.replace('.', '/') + '.lean')
anchors = {'bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py',
           'lakefile.toml', 'lake-manifest.json', 'lean-toolchain', 'bench/adapter_common.py'}
assert seen | anchors == set(inputs) and sorted(external) == after['external_imports']
for field, count in [('all_twenty_frozen_source_sha256', 20), ('audited_ttir_sha256', 13)]:
    assert len(after[field]) == count
    for path, h in after[field].items():
        check(path, h)
assert after['frozen_receipts'] == receipt['prior_frozen_receipts']
for path, entry in after['frozen_receipts'].items():
    check(Path(path) / 'evidence/receipt.json', entry['receipt'])
    check(Path(path) / 'evidence/scope-review.json', entry['scope'])
check(after['inherited_route_review']['path'], after['inherited_route_review']['sha256'])

rows = read(PKG / 'evidence/verification-results.json')
history = read(PKG / 'evidence/verification-attempts.json')
assert history[-6:] == rows and len(history) == 23
assert sum(row['exit_code'] != 0 for row in history) == 10
assert history[0]['exit_code'] == -15
for row in history:
    check(PKG / row['log'], row['log_sha256'])
    check(PKG / row['source_snapshot'], row['snapshot_sha256'])
    assert row['source_sha256_before'] == row['source_sha256_after'] == row['snapshot_sha256']
logs = []
for row in rows:
    assert row['exit_code'] == 0
    check(PKG / row['module'], row['source_sha256_after'])
    text = (PKG / row['log']).read_text()
    assert not re.search(r'\b(error|sorryAx|DISALLOWED)\b|warning:', text)
    logs.append(text)
assert sum(t.count('axiom footprint') for t in logs) == 40
assert sum(t.count('non-circular ✓') for t in logs) == 5
assert sum((PKG / rel).read_text().count('#auditModuleAxioms') for rel in receipt['modules']) == 6

review = {
    'schema': 1, 'kind': 'relu_source_coupled_block_grid_independent_review',
    'reviewer': 'root', 'status': 'accepted_partial_source_coupling_extension',
    'receipt_sha256': sha(PKG / 'evidence/receipt.json'),
    'scope_sha256': sha(PKG / 'evidence/scope-review.json'),
    'checker_path': str(Path(__file__).resolve()), 'checker_sha256': sha(__file__),
    'reviewed_module_sha256': receipt['modules'],
    'reviewed_input_sha256': dict(sorted(inputs.items())),
    'checks': {'artifact_hashes': len(receipt['artifacts']), 'dependencies': 190,
               'project_oleans': 179, 'predecessor_sources': 20, 'ttir_hashes': 13,
               'retained_compile_attempts': 23, 'retained_failures': 10,
               'final_successes': 6, 'named_axiom_checks': 40, 'noncircular_checks': 5,
               'module_audits': 6, 'warnings': 0, 'independent_lean_rerun': False},
    'semantic_review': [
        'All six Lean modules and producer verification/freeze/finalization code were read in full. SourceStages selects the exact seven statements of unchanged TypedValue.source; Valid retains seven actual Structural.step equations and derives original exec success.',
        'Checked loads consume the actual offset/mask registers and populate x. Inactive lanes preserve incoming State.undef; the original load has no other=0. The actual result statement runs for every pid before the extracted source gate.',
        'Physical stores use actual result/offset/mask registers and retain partial byte state on failure; the original logical last statement is separately executed and proved to agree. Unavailable first active input bytes fail before result/gate for every pid.',
        'Complete whole-program permutations pass preceding logical and byte state to each pid and retain every source trace. Final memory equals independent frozen wordSpec/byteGrid/byteSpec and preserves LogicalView, permissions and the output tail.',
        'Single-block initial input views allow aliases after loads. Grid composition requires distinct logical names and nonoverlapping full input/output byte intervals. Read permissions cover N inputs; write permissions cover only min(N,1024) fp32 outputs. Latent output values do not impose output-read permission.',
        'No new dtype profiles or numerical callbacks are introduced. The same nine integer/bool profiles and three pure total callback assumptions remain. New contracts are not automatically registered by the separate three-declaration exact numerical adapter.'
    ],
    'proof_defects': [], 'whole_original_program_accepted': False,
    'original_program_completion_increment': 0,
    'remaining_scope': scope['structural_remaining'],
    'numeric_validation': 'pending; no native numerical validation or GPU run claimed',
    'validation_boundary': 'Checks original producer evidence without rewriting it. Registered integration build and current source audit are still required.'
}
OUT.write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps({'review': str(OUT), 'sha256': sha(OUT), 'checks': review['checks']}, indent=2))
