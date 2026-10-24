"""Exercise proposal integration without modifying repository or claiming review."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

root=Path(__file__).resolve().parent
repo=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(repo/'bench'))
from adapter_common import canonical
from exact_scope_gate import validate_exact_case_links
from authority_adapter import extend_authority
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
delta=load(root/'evidence/numerical-authority-delta.json')
modules=next(iter(delta['provenance'].values()))['lean_source_sha256']
dependencies=load(root/'evidence/dependencies-before.json')
oldqueue=load(repo/'bench/tritonbench_g/deferred_numerical_experiments.json')
previous={'rules':[r for r in oldqueue['rules'] if r['status']=='pending_numeric_validation'],
  'provenance':oldqueue['exact_binding_provenance'],'requirements':oldqueue['exact_callback_requirements'],
  'input_sha256':oldqueue['exact_numerical_inputs'],'summary':oldqueue['exact_numerical_summary']}
baseline=deepcopy(previous)
with tempfile.TemporaryDirectory(prefix='adapter-smoke-',dir=root/'evidence') as tmp:
    fixture=Path(tmp);package=fixture/'checkpoint';package.mkdir()
    # Temporary integration layout, with explicitly recorded .lean.txt mapping.
    for mod in dependencies['project']:
        rel=mod.replace('.','/')+'.lean'
        p=fixture/rel;p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(repo/rel)
    for rel in [*modules,'bench/tritonbench_g/quant_transpose_kernel/quant_transpose_kernel.py']:
        p=fixture/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.symlink_to(root/rel if rel in modules else repo/rel)
    artifacts={'evidence/numerical-authority-delta.json':sha(root/'evidence/numerical-authority-delta.json'),
      'evidence/dependencies-before.json':sha(root/'evidence/dependencies-before.json')}
    for rel in artifacts:
        p=package/rel;p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(root/rel)
    for rel in modules:
        p=package/(rel+'.txt');p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(root/rel)
    receipt={'frozen':True,'modules':modules,'checkpoint_module_source_paths':{f:f+'.txt' for f in modules},'artifacts':artifacts}
    rp=package/'evidence/receipt.json';rp.write_text(json.dumps(receipt))
    h=sha(rp)
    extended=extend_authority(fixture,previous,'checkpoint',h)
    assert previous==baseline
    assert len(extended['provenance'])==len(previous['provenance'])+140
    assert all(k.startswith('repo:') for k in set(extended['input_sha256'])-set(previous['input_sha256']))
    conflicting=deepcopy(previous)
    conflicting['input_sha256']['repo:'+next(iter(modules))]='0'*64
    try:extend_authority(fixture,conflicting,'checkpoint',h)
    except ValueError as e:assert 'conflicting preexisting exact input' in str(e)
    else:raise AssertionError('conflicting preexisting input hash was accepted')
    rejected=0
    for args in [('0'*64,'checkpoint_lean_txt'),(h,'producer_lean')]:
        try:extend_authority(fixture,previous,'checkpoint',args[0],args[1])
        except (ValueError,OSError):rejected+=1
    assert rejected==2

table=delta['requirements'][0]
proof=(table['declaration']['file'],table['declaration']['declaration'])
exact={proof:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in table['cases']}}
links={}
for case in table['cases']:
    for r in case['required_binding_refs']:
        links.setdefault(r['callback'],set()).add((r['assumption_id'],r['scope_binding_id']))
missing_rejected=0
for case,row in exact[proof].items():
    applicable={(r['assumption_id'],r['scope_binding_id']) for r in row['required_binding_refs']}
    validate_exact_case_links(exact,proof,case,applicable,links)
    for r in row['required_binding_refs']:
        try:validate_exact_case_links(exact,proof,case,applicable-{(r['assumption_id'],r['scope_binding_id'])},links)
        except ValueError:missing_rejected+=1
        else:raise AssertionError('missing exact port was accepted')
assert missing_rejected==140
result={'status':'passed','adapter_layout':'explicit .lean to .lean.txt checkpoint mapping',
  'old_authority_preserved':True,'wrong_receipt_and_wrong_mapping_rejected':2,
  'conflicting_preexisting_input_rejected':True,'new_inputs_use_repo_prefix':True,
  'case_links_checked':735,'omitted_port_mutations_rejected':140,
  'source':'temporary synthetic receipt layout, not a root review or final integration',
  'global_exact_queue_equality':'unchanged; full root integration remains pending'}
(root/'evidence/adapter-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
