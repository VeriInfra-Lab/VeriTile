import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib, json
W = Path(__file__).resolve().parent
R = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0, str(R/'bench'))
from adapter_common import canonical
load = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest = lambda x: hashlib.sha256(canonical(x)).hexdigest()
old = load(W/'prior-v1/evidence/numerical-authority-delta.json')
new = load(W/'evidence/numerical-authority-delta.json')
def key(c):
    v = c['configuration']
    return (c['dtype_profile']['inputs']['x'], v['mode'], v['block_class'], v['size_class'], v['outcome'])
def index(delta):
    cases = delta['requirements'][0]['cases']
    result = {key(c): c for c in cases}
    assert len(result) == len(cases)
    return result
oi, ni = index(old), index(new)
assert len(oi) == 4648 and len(ni) == 4674 and set(oi) <= set(ni)
addition_keys = sorted(set(ni)-set(oi))
assert len(addition_keys) == 26 and all(k[2:] == ('other_valid', 'i32_minimum', 'grid_parse_rejected') for k in addition_keys)
mapping = []
ports = []
for k, o in oi.items():
    n = ni[k]
    assert o['required_callbacks'] == n['required_callbacks']
    assert len(o['required_binding_refs']) == len(n['required_binding_refs'])
    mapping.append({'key': list(k), 'old_case_sha256': digest(o), 'new_case_sha256': digest(n)})
    for a, b in zip(o['required_binding_refs'], n['required_binding_refs']):
        assert a['assumption_id'] == b['assumption_id'] and a['callback'] == b['callback']
        op = old['provenance'][a['assumption_id']+'::'+a['scope_binding_id']]
        np = new['provenance'][b['assumption_id']+'::'+b['scope_binding_id']]
        assert op['port'] == np['port'] and op['scalar_domain'] == np['scalar_domain']
        ports.append({'key': list(k), 'old_binding_ref': a, 'new_binding_ref': b,
                      'scalar_operation_unchanged': True,
                      'old_provenance_sha256': digest(op), 'new_provenance_sha256': digest(np)})
assert len(ports) == 64
lineage = load(W/'evidence/revision-lineage.json')
prior = load(W/lineage['prior_receipt'])
assert sha(W/lineage['prior_receipt']) == lineage['prior_receipt_sha256']
for source, path in lineage['prior_modules'].items():
    assert sha(W/path) == prior['modules'][source]
for source, path in lineage['prior_artifacts'].items():
    assert sha(W/path) == prior['artifacts'][source]
peer = load(W/'peer-review/review-receipt.json')
assert sha(W/'peer-review/review-receipt.json') == lineage['peer_receipt_sha256']
assert sha(W/'peer-review/peer-review.json') == lineage['peer_review_sha256']
for source, path in lineage['peer_evidence'].items():
    assert sha(W/path) == peer['artifacts'][source]
report = {'status': 'passed', 'revision': 2,
          'old_receipt_sha256': lineage['prior_receipt_sha256'],
          'old_public_statement': old['requirements'][0]['declaration'],
          'new_public_statement': new['requirements'][0]['declaration'],
          'old_delta_sha256': sha(W/'prior-v1/evidence/numerical-authority-delta.json'),
          'new_delta_sha256': sha(W/'evidence/numerical-authority-delta.json'),
          'old_cases': len(oi), 'new_cases': len(ni), 'mapped_cases': len(mapping),
          'added_cases': [{'key': list(k), 'case_sha256': digest(ni[k]),
                           'required_binding_refs': ni[k]['required_binding_refs']} for k in addition_keys],
          'mapped_numerical_ports': len(ports), 'original_frozen_artifacts_verified': len(prior['artifacts']),
          'original_frozen_sources_verified': len(prior['modules']),
          'copied_peer_artifacts_verified': len(lineage['peer_evidence']),
          'case_mapping': mapping, 'numerical_port_mapping': ports,
          'acceptance_increment': 0, 'prior_review_status': 'preserved_blocked'}
(W/'evidence/case-mapping-v1-v2.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k not in ['case_mapping', 'numerical_port_mapping', 'added_cases']}))
