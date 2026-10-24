from pathlib import Path
import sys,hashlib,json,collections
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/triton-mul2-finite-acceptance-revision2-wip');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from exact_scope_gate import validate_exact_case_links
from check_original_program_scope import validate_case,validate_deferrals,declaration_scope
load=lambda p:json.loads(p.read_text())
delta=load(P/'evidence/numerical-authority-delta.json');table=delta['requirements'][0];cases=table['cases'];plain=load(P/'evidence/scope-cases.json');row=load(P/'evidence/scope-row-proposal.json');decl=load(P/'evidence/declaration-proposal.json')
ref=table['declaration'];actual=declarations(P/'after'/ref['file'])[ref['declaration']];assert actual['kind']=='specification' and actual['statement_sha256']==ref['statement_sha256']
key=(ref['file'],ref['declaration']); exact={key:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in cases}}
assert len(exact[key])==4674 and len(plain)==4674
assert set(exact[key])=={canonical(c) for c in plain}
assert sha(canonical(sorted(plain,key=canonical)))==table['supported_cases_sha256']
for c in plain:validate_case(c)
declaration_scope(decl,row['source_inventory']);validate_deferrals('triton_mul2',row['deferred_numerical_assumptions'],{'rules':delta['rules']})
links={}; numeric=0;omissions=0; seenrefs=set()
for c in cases:
 refs=c['required_binding_refs'];dtype=c['dtype_profile']['inputs']['x'];conf=c['configuration']
 expected=dtype in ['fp16','bf16','fp32','fp64'] and conf['outcome']=='returned_active'
 assert bool(refs)==expected,(dtype,conf)
 assert len(refs)==int(expected)
 if expected:numeric+=1
 for ref in refs:
  pair=(ref['assumption_id'],ref['scope_binding_id']);seenrefs.add(pair);links.setdefault(ref['callback'],set()).add(pair)
assert numeric==64 and len(seenrefs)==64 and len(delta['provenance'])==64
for ident,entry in delta['provenance'].items():
 assert ident==entry['assumption_id']+'::'+entry['scope_binding_id']
 assert (entry['assumption_id'],entry['scope_binding_id']) in seenrefs
 assert entry['callback_declaration']=='TritonMul2Finite.Numerics.multiply'
 assert entry['status']=='explicitly_deferred_unvalidated' and entry['validation_evidence'] is None
 for fld in ['raw_ttir','frontend_receipt','producer_scalar_inventory']:
  rr=entry[fld];assert hashlib.sha256((P/rr['path']).read_bytes()).hexdigest()==rr['sha256']
for case,c in exact[key].items():
 refs={(r['assumption_id'],r['scope_binding_id']) for r in c['required_binding_refs']}
 validate_exact_case_links(exact,key,case,refs,links)
 for missing in refs:
  try:validate_exact_case_links(exact,key,case,refs-{missing},links)
  except ValueError:omissions+=1
  else:raise AssertionError('missing port accepted')
assert omissions==64
# Independently compare old and revised logical case keys, without trusting producer mapping.
old=load(Path('/tmp/veritile-standardization-20261015/triton-mul2-finite-acceptance-wip/evidence/numerical-authority-delta.json'))
def casekey(c):
 cfg=c['configuration'];return (c['dtype_profile']['inputs']['x'],cfg['mode'],cfg['block_class'],cfg['size_class'],cfg['outcome'])
a={casekey(c):c for c in old['requirements'][0]['cases']};z={casekey(c):c for c in cases}
assert len(a)==4648 and len(z)==4674 and set(a)<=set(z)
added=set(z)-set(a)
expected={(dtype,mode,'other_valid','i32_minimum','grid_parse_rejected') for mode in ['outplace','inplace'] for dtype in ['bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','fp16','bf16','fp32','fp64']}
assert added==expected
for k in a:assert a[k]['required_callbacks']==z[k]['required_callbacks'] and len(a[k]['required_binding_refs'])==len(z[k]['required_binding_refs'])
assert all(not z[k]['required_binding_refs'] for k in added)
result={'status':'passed','cases_checked':len(cases),'numeric_cases':numeric,'exact_ports':len(seenrefs),'omitted_port_mutations_rejected':omissions,'public_statement_sha256':actual['statement_sha256'],'zero_numeric_empty_cases':sum(c['configuration']['size_class']=='empty' for c in cases),'zero_numeric_i32_minimum_cases':sum(c['configuration']['size_class']=='i32_minimum' for c in cases),'old_cases_preserved':len(a),'new_cases':sorted(added),'new_case_count':len(added),'old_case_numerical_callback_requirements_preserved':True,'gpu_executions':0}
(W/'case-and-port-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='new_cases'}))
