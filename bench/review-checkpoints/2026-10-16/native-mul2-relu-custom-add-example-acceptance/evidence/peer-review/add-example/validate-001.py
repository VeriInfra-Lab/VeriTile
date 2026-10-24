from pathlib import Path
import sys,hashlib,json,collections,importlib.util
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/add-example-finite-acceptance-wip');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from exact_scope_gate import validate_exact_case_links
from check_original_program_scope import validate_case,validate_deferrals,declaration_scope
spec=importlib.util.spec_from_file_location('custom_scope_helpers',P/'scope_helpers.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
load=lambda p:json.loads(p.read_text());hf=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=load(P/'evidence/numerical-authority-delta.json');table=d['requirements'][0];cases=table['cases'];plain=load(P/'evidence/scope-cases.json');row=load(P/'evidence/scope-row-proposal.json');decl=load(P/'evidence/declaration-proposal.json');receipt=load(P/'evidence/receipt.json')
ref=table['declaration'];actual=declarations(P/ref['file'])[ref['declaration']];assert actual['kind']=='specification' and actual['statement_sha256']==ref['statement_sha256']
key=(ref['file'],ref['declaration']);exact={key:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in cases}}
assert len(exact[key])==len(cases)==len(plain)==1700
assert set(exact[key])=={canonical(c) for c in plain}
assert sha(canonical(sorted(plain,key=canonical)))==table['supported_cases_sha256']
for c in plain:validate_case(c)
declaration_scope(decl,row['source_inventory']);validate_deferrals('add_example',row['deferred_numerical_assumptions'],{'rules':d['rules']})
profiles={(h.dtype(p['x_dtype']),h.dtype(p['y_dtype'])):p for p in load(P/'evidence/inherited/typed-host-numerical-proposal.json')['profiles']}
native=P/'evidence/native-attempts/20261015T215117570205Z/receipt.json';native_rows={(h.dtype(r['x']),h.dtype(r['y'])):r for r in load(native.with_name('records.json'))};backend_rows={(h.dtype(r['x']),h.dtype(r['y'])):r for r in load(P/'evidence/backend-attempts/20261015T215542684768Z/records.json')}
links={};seen=set();omissions=0;groups=collections.defaultdict(set)
for c in cases:
 cfg=c['configuration'];pair=tuple(c['dtype_profile']['inputs'][x] for x in ['x','y']);refs=c['required_binding_refs'];outcome=cfg['outcome'];groups[pair].add(outcome)
 assert cfg['BLOCK_SIZE']==4 and cfg['entrypoint']=='add_wrapper'
 selected=profiles[pair]['operations'] if outcome=='returned_defined' else []
 assert len(refs)==len(selected)
 assert c['required_callbacks']==sorted({rr['callback'] for rr in refs})
 for i,(rr,op) in enumerate(zip(refs,selected)):
  pairref=(rr['assumption_id'],rr['scope_binding_id']);seen.add(pairref);links.setdefault(rr['callback'],set()).add(pairref);port=d['provenance']['::'.join(pairref)]
  assert port['port']['name']==op['port'] and port['port']['ordinal']==i and port['port']['operation']==op['operation']
  assert port['callback_declaration']==rr['callback']==h.callback(op['port']) and port['port']['scalar_configuration']==op['configuration']
  assert port['configuration']==cfg and port['public_statement']==ref and port['lean_source_sha256']==receipt['modules']
  assert port['status']=='explicitly_deferred_unvalidated' and port['validation_evidence'] is None and not port['native_execution']
  for field in ['raw_ttir','producer_scalar_inventory']:assert hf(P/port[field]['path'])==port[field]['sha256']
  for field in ['original_source','scalar_source']:assert hf(R/port[field]['path'])==port[field]['sha256']
  for field in ['execution_source']:assert hf(R/port[field]['module'])==port[field]['sha256']
 extent=cfg['n_elements'];limit=(2**31-1)*4
 expected_extent=({'equals':0} if outcome=='returned_empty' else {'native_launchable':True,'residue_mod_2pow32':2**31} if outcome=='returned_all_masked' else {'lower_exclusive':limit} if outcome=='native_grid_parse_rejected' else {'lower_inclusive':1,'upper_inclusive':limit,'residue_mod_2pow32_not_equal':2**31} if outcome in ['returned_defined','source_applicability_observation'] else {'lower_inclusive':0,'upper_exclusive':'unbounded_actual_Torch_numel'})
 assert extent==expected_extent
for pair,outcomes in groups.items():
 f=native_rows[pair];b=backend_rows.get(pair)
 if f['status']=='frontend_generated':terminal='admitted' if b['status']=='compiled' else 'backend_rejected'
 elif f['error_type']=='KeyError':terminal='binder_rejected'
 elif 'not supported in this architecture' in f['error']:terminal='signature_rejected'
 else:terminal='addition_rejected'
 expected={'host_or_native_setup_failure',terminal} if terminal!='admitted' else {'host_or_native_setup_failure','native_grid_parse_rejected','returned_empty','returned_all_masked','source_applicability_observation','returned_defined'}
 assert outcomes==expected
assert len(groups)==484 and len(seen)==len(d['provenance'])==296 and len(d['rules'])==37
for case,c in exact[key].items():
 refs={(r['assumption_id'],r['scope_binding_id']) for r in c['required_binding_refs']};validate_exact_case_links(exact,key,case,refs,links)
 for missing in refs:
  try:validate_exact_case_links(exact,key,case,refs-{missing},links)
  except ValueError:omissions+=1
  else:raise AssertionError('missing port accepted')
assert omissions==296
result={'status':'passed','scope_cases':1700,'dtype_pairs':484,'numeric_ports':296,'numeric_families':37,'omitted_port_mutations_rejected':omissions,'zero_numeric_empty_cases':sum(c['configuration']['outcome']=='returned_empty' for c in cases),'zero_numeric_all_masked_cases':sum(c['configuration']['outcome']=='returned_all_masked' for c in cases),'public_statement_sha256':ref['statement_sha256'],'current_module_count':len(receipt['modules']),'GPU_launches':0}
(W/'case-and-port-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
