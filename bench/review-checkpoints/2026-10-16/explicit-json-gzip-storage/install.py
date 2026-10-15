from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
R = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P = Path('/tmp/veritile-standardization-20261015/json-gzip-storage-proposal-wip')
B = R / 'bench/review-checkpoints/2026-10-16/explicit-json-gzip-storage'
EXPECTED = '48b2f06eb4b0194e8a13c7c03fda82e9ca50cf88f1a4fd9701fc854ad9b6ef0e'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

assert not B.exists()
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=R)
assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=R).decode().strip() == 'fp-examples'
base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=R).decode().strip()
assert sha(P / 'evidence/receipt.json') == EXPECTED
receipt = json.loads((P / 'evidence/receipt.json').read_text())
for rel, digest in receipt['artifacts'].items():
    assert sha(P / rel) == digest, rel
for rel, digest in receipt['modified_source_before_sha256'].items():
    assert sha(R / rel) == digest, rel
for rel, digest in receipt['candidate_files'].items():
    assert sha(P / 'after' / rel) == digest, rel
    if rel not in receipt['modified_source_before_sha256']:
        assert not (R / rel).exists(), rel
assert receipt['final_scope_comparison']['counts'] == {
    'not_accepted': 179, 'verified': 1, 'verified_with_deferred_numerics': 3}

# Store every frozen producer artifact losslessly, avoiding duplicate expanded
# snapshots. Member hashes remain the original producer's hashes.
B.mkdir(parents=True)
members = dict(receipt['artifacts'])
members['evidence/receipt.json'] = EXPECTED
with zipfile.ZipFile(B / 'producer-evidence.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for rel, digest in sorted(members.items()):
        data = (P / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
with zipfile.ZipFile(B / 'producer-evidence.zip') as archive:
    assert sorted(archive.namelist()) == sorted(members)
    for rel, digest in members.items():
        assert hashlib.sha256(archive.read(rel)).hexdigest() == digest, rel
save(B / 'archive-index.json', {'schema': 1, 'archive_sha256': sha(B / 'producer-evidence.zip'),
                              'member_sha256': members, 'producer_receipt_sha256': EXPECTED})
shutil.copyfile(P / 'evidence/receipt.json', B / 'producer-receipt.json')
shutil.copyfile(Path(__file__), B / 'install.py')
for rel in receipt['candidate_files']:
    shutil.copyfile(P / 'after' / rel, R / rel)

modules = ['bench.tests.test_json_storage', 'bench.tests.test_gzip_source_audit',
           'bench.tests.test_current_source_adapters', 'bench.tests.test_current_source_audit',
           'bench.tests.test_current_value_metadata', 'bench.tests.test_original_program_scope',
           'scripts.test_audit_shards']
env = dict(os.environ, PYTHONPATH=str(R / 'bench/tests'), PYTHONDONTWRITEBYTECODE='1')
with (B / 'installed-tests.log').open('w') as stream:
    run = subprocess.run([sys.executable, '-B', '-m', 'unittest', *modules], cwd=R, env=env,
                         stdout=stream, stderr=subprocess.STDOUT)
assert run.returncode == 0, 'Installed tests failed; retain log and source before repair'
log = (B / 'installed-tests.log').read_text()
assert 'Ran 228 tests' in log and log.rstrip().endswith('OK'), log[-2000:]
for rel, digest in receipt['candidate_files'].items():
    assert sha(R / rel) == digest, rel

save(B / 'root-review.json', {
    'schema': 1, 'reviewer': 'root', 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'base_commit': base, 'producer_receipt_sha256': EXPECTED,
    'status': 'reviewed_and_installed_storage_support', 'candidate_files': receipt['candidate_files'],
    'review': {
        'implementation': 'Read complete production diff and all new/changed tests. Only exact case-sensitive .json.gz paths opt into gzip; plain JSON formatting and duplicate-key validation are preserved. UTF8 gzip has no embedded filename and timestamp zero. Full-stream reads validate trailers before parsing. No fallback or automatic migration is added.',
        'authority': 'Canonical JSON hashes, raw file hashes, path validation and schema checks are unchanged. Metadata-only compressed-byte mutation invalidates raw-byte receipts; malformed or truncated streams fail closed.',
        'real_data': 'Verified all 203 frozen producer artifacts and the frozen receipt. Producer compared actual catalog, supplement, root review and ledger objects/canonical hashes; final plain/gzip scope outputs were byte-identical at one verified, three deferred,179 pending. Preserved stale-helper and harness failures remain in the lossless archive.',
        'validation': 'Root ran228 tests against the installed candidate: storage, compressed audits and bound reviews, plain source adapters/audits/value metadata/scope, and shard dispatch. Stubbed comparator fixtures do not establish comparator or GPU correctness.',
        'scope': 'This commit adds storage support only. It neither rewrites historical evidence nor migrates any active catalog/ledger; future integration must use explicit paths and normal current gates. Accepted programs remain4/183.'},
    'installed_test_count': 228, 'installed_test_exit_code': run.returncode,
    'installed_test_log_sha256': sha(B / 'installed-tests.log'),
    'archive_sha256': sha(B / 'producer-evidence.zip'), 'accepted_programs': 4,
    'numerical_conformance_validated': False, 'native_gpu_execution': False,
    'official_comparator_passed': False})
(B / 'README.md').write_text('''# Explicit gzip JSON storage support

The shared JSON helper now supports explicit `.json.gz` paths while retaining
plain JSON behavior. Full-stream gzip validation, duplicate-key rejection,
canonical object hashing and raw-byte receipt binding remain enforced.

The root reviewed the production diff and tests, verified all203 frozen producer
artifacts, and ran228 installed-source tests successfully. The producer's actual
plain/gzip scope comparison retained one verified plus three scopes with deferred
numerics and179 pending. This storage change adds no accepted program.

`producer-evidence.zip` losslessly retains every producer-receipt artifact and
the receipt itself. `archive-index.json` binds every uncompressed member hash
and the actual archive bytes. Initial failed attempts remain included. The
snapshots are evidence only; no active ledger or historical file is migrated.
Test comparator stubs do not establish official comparator or GPU correctness.
''')
save(B / 'validation.json', {'schema': 1, 'base_commit': base, 'status': 'passed',
    'candidate_files': receipt['candidate_files'],
    'checkpoint_files': {p.name: sha(p) for p in B.iterdir() if p.is_file()},
    'installed_tests': 228, 'accepted_programs': 4})
print(json.dumps({'checkpoint': str(B), 'tests': 228, 'archive_bytes': (B / 'producer-evidence.zip').stat().st_size,
                  'root_review_sha256': sha(B / 'root-review.json')}, indent=2))
