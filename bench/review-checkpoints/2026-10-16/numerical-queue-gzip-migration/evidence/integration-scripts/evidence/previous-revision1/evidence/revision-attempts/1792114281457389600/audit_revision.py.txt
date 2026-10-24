"""Check only the independently requested revision; preserve previous replay evidence."""
from pathlib import Path
import ast
import difflib
import json
import sys

sys.dont_write_bytecode = True
W = Path(__file__).resolve().parent
R = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P = W.parent / 'queue-gzip-integration-scripts-wip'
sys.path.insert(0, str(W))
import integration as m

parent_hash = '18f8b40a7718ef958edad7ef67dad8bbe919732ec1572c9a600e8cf4a7ed29d4'
parent = m.verify_script_package(P, parent_hash)
preserved = m.verify_script_package(W / 'evidence/previous-candidate', parent_hash)
assert preserved == parent
for rel, row in parent['proposed_installed_files'].items():
    assert m.file_hash(W / 'after' / rel) == row['after']
    assert (m.file_hash(R / rel) if (R / rel).exists() else None) == row['before']
for rel, h in {**parent['unchanged_consumers'], **parent['policy_inputs']}.items():
    assert m.file_hash(R / rel) == h

changed_functions = {'verify_candidate', 'read_plan', 'plan_stage', 'install_stage', 'verify_installed', 'check_stage'}
new_functions = {'external_run', 'verify_script_package', 'copy_script_package'}
def bodies(path):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
before = bodies(P / 'integration.py')
after = bodies(W / 'integration.py')
assert set(after) - set(before) == new_functions
assert set(before) - set(after) == set()
assert {name for name in before if before[name] != after[name]} == changed_functions
def unchanged_ast(path):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef) or node.name not in changed_functions | new_functions]
    return ast.dump(tree, include_attributes=False)
assert unchanged_ast(P / 'integration.py') == unchanged_ast(W / 'integration.py')
def test_class(path, name):
    return ast.dump(next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == name), include_attributes=False)
assert test_class(P / 'test_integration.py', 'IntegrationStorageTests') == test_class(W / 'test_integration.py', 'IntegrationStorageTests')

patch = []
for name in ['integration.py', 'test_integration.py', 'PLAN.md', 'README.md', 'run_checks.py', 'freeze.py', 'audit_revision.py', 'REVISION.md']:
    old = (P / name).read_text().splitlines(True) if (P / name).is_file() else []
    new = (W / name).read_text().splitlines(True)
    patch.extend(difflib.unified_diff(old, new, fromfile='previous/' + name, tofile='revision1/' + name))
(W / 'evidence/revision.patch').write_text(''.join(patch))
data = {
    'status': 'passed', 'previous_receipt_sha256': parent_hash,
    'previous_artifacts_reverified': len(parent['artifacts']),
    'preserved_previous_artifacts_reverified': len(preserved['artifacts']),
    'six_installed_candidates_byte_identical': True,
    'source_plan_candidate_patch_and_previous_checks_retained': all(m.file_hash(W / rel) == parent['artifacts'][rel] for rel in ['evidence/source-plan.json', 'evidence/candidate.patch', 'evidence/candidate-audit.json', 'evidence/checks.json']),
    'unchanged_consumers_and_policy_pins_reverified': True,
    'changed_integration_functions': sorted(changed_functions), 'new_integration_functions': sorted(new_functions),
    'all_other_integration_ast_unchanged': True, 'original_eight_integration_tests_unchanged': True,
    'baseline_validation_reuse_added': False, 'repository_modified': False,
    'live_plan_or_migration_run': False, 'acceptance_increment': 0,
}
m.write(W / 'evidence/revision-audit.json', data)
print(json.dumps(data))
