"""Exercise a synthetic integration fixture; this never supplies real approval."""
import sys
sys.dont_write_bytecode = True
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

root = Path(__file__).resolve().parent
repo = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0, str(repo / 'bench'))
# Exact committed pre-quant authority supplied by root; never mutate repository.
import importlib.util
baseline_path = root / 'evidence/pre-quant-baseline/reviewed_numerical_extensions.py'
baseline_spec = importlib.util.spec_from_file_location('reviewed_numerical_extensions', baseline_path)
baseline_module = importlib.util.module_from_spec(baseline_spec)
sys.modules['reviewed_numerical_extensions'] = baseline_module
baseline_spec.loader.exec_module(baseline_module)

from adapter_common import canonical
from exact_scope_gate import validate_exact_case_links
from exact_numerical_bindings import collect, extend_queue, validate_exact_queue
from authority_adapter import extend_authority, project_dependency_inputs
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_text())
dump = lambda p, d: Path(p).write_text(json.dumps(d, indent=2) + '\n')
delta = load(root / 'evidence/numerical-authority-delta.json')
modules = next(iter(delta['provenance'].values()))['lean_source_sha256']
snapshots = delta['dependency_snapshots']
project_inputs = project_dependency_inputs(root, snapshots)
port_count = len(delta['provenance'])
case_count = sum(len(t['cases']) for t in delta['requirements'])
assert delta['summary']['scope_bindings'] == port_count
assert delta['summary']['declaration_cases'] == case_count
assert delta['summary']['scalar_families'] == len(delta['rules'])

# Producer-local verification of every before-first-use object. These paths
# remain historical evidence and never become permanent CI authority inputs.
project_objects = {}
external_objects = {}
for reference in snapshots:
    snap = load(root / reference['path'])
    assert sha(root / reference['path']) == reference['sha256']
    for module, entry in snap.get('project', {}).items():
        assert sha(Path(entry['source'])) == entry['sha256'], module
        assert sha(Path(entry['olean'])) == entry['olean_sha256'], module
        project_objects[module] = {'source_sha256':entry['sha256'], 'object_sha256':entry['olean_sha256']}
    for module, entry in snap.get('external_objects', {}).items():
        assert sha(Path(entry['olean'])) == entry['sha256'], module
        external_objects[module] = entry['sha256']
    for path, expected in snap.get('input_sha256', {}).items():
        assert sha(Path(path)) == expected, path
object_check = {'status':'passed', 'snapshots':snapshots, 'project_sources_and_objects':project_objects,
    'external_objects':external_objects, 'queue_authority':'Only portable repository sources and retained checkpoint evidence; no local build-object paths.'}
dump(root / 'evidence/adapter-dependency-object-check.json', object_check)
legacy, previous, _ = collect(repo)
baseline = deepcopy(previous)
baseline_queue = load(root / 'evidence/pre-quant-baseline/deferred_numerical_experiments.json')
assert len(extend_queue(legacy, previous)['rules']) == 100
assert len(previous['rules']) == 89
assert len(previous['requirements']) == 6
assert canonical(extend_queue(legacy, previous)['rules'][:100]) == canonical(baseline_queue['rules'][:100])
baseline_receipt = load(root / 'evidence/pre-quant-baseline/receipt.json')
assert sha(baseline_path) == baseline_receipt['artifacts']['reviewed_numerical_extensions.py']
rejected = []
with tempfile.TemporaryDirectory(prefix='adapter-smoke-', dir=root / 'evidence') as tmp:
    fixture = Path(tmp)
    package = fixture / 'checkpoint'
    package.mkdir()
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in project_inputs:
        copy(repo / relative, fixture / relative)
    for relative in modules:
        copy(root / relative, fixture / relative)
        copy(root / relative, package / (relative + '.txt'))
    original = delta['original_source']
    copy(repo / original['path'], fixture / original['path'])
    retained = {'evidence/numerical-authority-delta.json':sha(root / 'evidence/numerical-authority-delta.json')}
    retained.update({r['path']:r['sha256'] for r in snapshots})
    for evidence in delta['native_evidence']:
        retained.update(evidence['retained_inputs'])
    for relative in retained:
        copy(root / relative, package / relative)
    captures = {r['path']:{'sha256':r['sha256'], 'verified_unchanged':True} for r in snapshots}
    after = {'captures':captures, 'project_sources_and_objects':len(project_objects),
        'external_objects':len(external_objects), 'all_unchanged':True}
    dump(package / 'evidence/dependencies-after.json', after)
    retained['evidence/dependencies-after.json'] = sha(package / 'evidence/dependencies-after.json')
    receipt = {'frozen':True, 'modules':modules,
        'checkpoint_module_source_paths':{p:p+'.txt' for p in modules}, 'artifacts':retained}
    rp = package / 'evidence/receipt.json'
    review_path = fixture / 'synthetic-root-review.json'
    review = {'reviewer':'root', 'status':'substantively_reviewed_for_finite_scope_integration',
        'synthetic_fixture_only':True, 'not_an_actual_review':True}
    def repin():
        dump(rp, receipt)
        review['producer_receipt_sha256'] = sha(rp)
        dump(review_path, review)
        return sha(rp), sha(review_path)
    rh, vh = repin()
    def extend(prior=previous, receipt_hash=None, review_hash=None):
        return extend_authority(fixture, prior, 'checkpoint', receipt_hash or rh,
            'synthetic-root-review.json', review_hash or vh)
    extended = extend()
    assert previous == baseline
    assert len(extended['provenance']) == len(previous['provenance']) + port_count
    new_inputs = set(extended['input_sha256']) - set(previous['input_sha256'])
    assert all(key.startswith('repo:') and '/.lake/' not in key and not key.startswith('repo:/') for key in new_inputs)
    proposed = extend_queue(legacy, extended)
    queue_check = validate_exact_queue(proposed, extended, legacy)
    def reject(name, action):
        try:
            action()
        except (ValueError, OSError, KeyError):
            rejected.append(name)
        else:
            raise AssertionError('tamper accepted: ' + name)
    reject('wrong_receipt', lambda: extend(receipt_hash='0'*64))
    reject('wrong_review', lambda: extend(review_hash='0'*64))
    conflicting = deepcopy(previous)
    conflicting['input_sha256']['repo:' + next(iter(modules))] = '0'*64
    reject('conflicting_preexisting_input', lambda: extend(prior=conflicting))
    tamper_paths = {
        'changed_checkpoint_module':package / (next(iter(modules)) + '.txt'),
        'changed_current_helper':fixture / next(iter(modules)),
        'changed_original_source':fixture / original['path'],
        'changed_legacy_source':fixture / next(iter(delta['provenance'].values()))['legacy_route_source']['path'],
    }
    for index, reference in enumerate(snapshots):
        tamper_paths['changed_dependency_snapshot_' + str(index)] = package / reference['path']
        for module in load(root / reference['path']).get('project', {}):
            tamper_paths['changed_project_dependency_' + str(index)] = fixture / (module.replace('.', '/') + '.lean')
            break
    for evidence in delta['native_evidence']:
        role = evidence['role']
        tamper_paths['changed_native_receipt_' + role] = package / evidence['receipt']['path']
        artifact = next(p for p in evidence['retained_inputs'] if p.endswith('.ttir'))
        tamper_paths['changed_native_ttir_' + role] = package / artifact
    for name, path in tamper_paths.items():
        saved = path.read_bytes()
        path.write_bytes(saved + b'\n')
        reject(name, extend)
        path.write_bytes(saved)
    # Re-pin intentionally altered fixture manifests to test semantic checks,
    # beyond the outer receipt checksum. None of these is an actual approval.
    saved_mapping = deepcopy(receipt['checkpoint_module_source_paths'])
    receipt['checkpoint_module_source_paths'][next(iter(modules))] = next(iter(modules))
    rh, vh = repin()
    reject('wrong_checkpoint_mapping', extend)
    receipt['checkpoint_module_source_paths'] = saved_mapping
    rh, vh = repin()
    review['reviewer'] = 'producer'
    rh, vh = repin()
    reject('producer_self_review', extend)
    review['reviewer'] = 'root'
    rh, vh = repin()
    review['producer_receipt_sha256'] = '0'*64
    dump(review_path, review)
    vh = sha(review_path)
    reject('review_for_other_receipt', extend)
    rh, vh = repin()
    retained_snapshot = snapshots[-1]['path']
    retained_hash = receipt['artifacts'].pop(retained_snapshot)
    rh, vh = repin()
    reject('omitted_dependency_snapshot_manifest', extend)
    receipt['artifacts'][retained_snapshot] = retained_hash
    rh, vh = repin()
    native_artifact = next(p for p in delta['native_evidence'][-1]['retained_inputs'] if p.endswith('.ttir'))
    native_hash = receipt['artifacts'].pop(native_artifact)
    rh, vh = repin()
    reject('omitted_native_artifact_manifest', extend)
    receipt['artifacts'][native_artifact] = native_hash
    rh, vh = repin()
    bad_queue = deepcopy(proposed)
    bad_queue['rules'][-1]['required_validation'] = 'tampered'
    reject('changed_whole_queue', lambda: validate_exact_queue(bad_queue, extended, legacy))
    payload = {k:v for k,v in proposed.items() if k != 'exact_numerical_inputs'}
    payload['exact_numerical_inputs_pending_independent_review'] = True
    dump(root / 'evidence/queue-payload-proposal.json', payload)

missing_rejected = 0
numeric_cases = 0
observation_cases = 0
for table in delta['requirements']:
    proof = (table['declaration']['file'], table['declaration']['declaration'])
    exact = {proof:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in table['cases']}}
    links = {}
    for case in table['cases']:
        for reference in case['required_binding_refs']:
            links.setdefault(reference['callback'], set()).add((reference['assumption_id'], reference['scope_binding_id']))
    for key, case in exact[proof].items():
        applicable = {(r['assumption_id'],r['scope_binding_id']) for r in case['required_binding_refs']}
        validate_exact_case_links(exact, proof, key, applicable, links)
        numeric_cases += bool(applicable)
        if case['configuration']['outcome'] == 'native_observation':
            observation_cases += 1
            assert not applicable and not case['required_callbacks']
        for omitted in applicable:
            reject_name = 'missing exact port'
            try:
                validate_exact_case_links(exact, proof, key, applicable - {omitted}, links)
            except ValueError:
                missing_rejected += 1
            else:
                raise AssertionError(reject_name + ' accepted')
assert missing_rejected == port_count
assert previous == baseline
result = {'status':'passed', 'synthetic_review_fixture_only':True, 'independent_review':'pending',
    'adapter_layout':'explicit .lean to .lean.txt checkpoint mapping', 'old_authority_preserved':True,
    'adapter_mutations_rejected':rejected, 'new_inputs_use_portable_repo_prefix':True,
    'case_links_checked':case_count, 'numeric_cases':numeric_cases, 'structural_observation_cases':observation_cases,
    'omitted_port_mutations_rejected':missing_rejected, 'unchanged_whole_queue_equality_test':queue_check,
    'dependency_snapshots_checked':len(snapshots), 'project_sources_and_objects_checked':len(project_objects),
    'external_objects_checked':len(external_objects), 'native_receipts_checked':len(delta['native_evidence']),
    'final_current_repository_integration':'not performed', 'acceptance_increment':0, 'baseline_commit':baseline_receipt['commit'], 'baseline_module_sha256':sha(baseline_path), 'first100_rules_equal_committed_queue':True}
dump(root / 'evidence/adapter-check.json', result)
print(json.dumps(result))
