from pathlib import Path
import sys,hashlib,json,collections,importlib.util
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/vector-custom-finite-acceptance-revision1-wip');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from exact_scope_gate import validate_exact_case_links
from check_original_program_scope import validate_case,validate_deferrals,declaration_scope
spec=importlib.util.spec_from_file_location('custom_scope_helpers',P/'scope_helpers.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
load=lambda p:json.loads(p.read_text());hf=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=load(P/'evidence/numerical-authority-delta.json');table=d['requirements'][0];cases=table['cases'];plain=load(P/'evidence/scope-cases.json');row=load(P/'evidence/scope-row-proposal.json');decl=load(P/'evidence/declaration-proposal.json');receipt=load(P/'evidence/receipt.json')
ref=table['declaration'];actual=declarations(P/'after'/ref['file'])[ref['declaration']];assert actual['kind']=='specification' and actual['statement_sha256']==ref['statement_sha256']
key=(ref['file'],ref['declaration']);exact={key:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in cases}}
assert len(exact[key])==len(cases)==len(plain)==4498
assert set(exact[key])=={canonical(c) for c in plain}
assert sha(canonical(sorted(plain,key=canonical)))==table['supported_cases_sha256']
for c in plain:validate_case(c)
declaration_scope(decl,row['source_inventory']);validate_deferrals('vector_addition_custom',row['deferred_numerical_assumptions'],{'rules':d['rules']})
profiles={(h.dtype(p['x_dtype']),h.dtype(p['y_dtype'])):p for p in load(P/'evidence/inherited/typed-host-numerical-proposal.json')['profiles']}
native=P/load(P/'evidence/native-latest.json')['receipt'];native_rows={(h.dtype(r['x']),h.dtype(r['y'])):r for r in load(native.with_name('records.json'))}
links={};seen=set();omissions=0;groups=collections.defaultdict(set)
for c in cases:
 cfg=c['configuration'];pair=tuple(c['dtype_profile']['inputs'][x] for x in ['x','y']);refs=c['required_binding_refs'];outcome=cfg['outcome'];groups[pair].add((outcome,cfg['native_observation']))
 assert cfg['BLOCK']==16 and cfg['entrypoint']=='custom_add'
 selected=profiles[pair]['operations'] if outcome=='returned_or_undefined_conversion' else []
 assert len(refs)==len(selected)
 assert c['required_callbacks']==sorted({rr['callback'] for rr in refs})
 for i,(rr,op) in enumerate(zip(refs,selected)):
  pairref=(rr['assumption_id'],rr['scope_binding_id']);seen.add(pairref);links.setdefault(rr['callback'],set()).add(pairref);port=d['provenance']['::'.join(pairref)]
  assert port['port']['name']==op['port'] and port['port']['ordinal']==i and port['port']['operation']==op['operation']
  assert port['callback_declaration']==rr['callback']==h.callback(op['port']) and port['port']['scalar_configuration']==op['configuration']
  assert port['configuration']==cfg and port['public_statement']==ref and port['lean_source_sha256']==receipt['modules']
  assert port['status']=='explicitly_deferred_unvalidated' and port['validation_evidence'] is None and not port['native_execution']
  for field in ['raw_ttir','frontend_receipt','producer_scalar_inventory']:assert hf(P/port[field]['path'])==port[field]['sha256']
  for field in ['original_source','scalar_source']:assert hf(R/port[field]['path'])==port[field]['sha256']
  for field in ['execution_source','host_source']:assert hf(R/port[field]['module'])==port[field]['sha256']
 assert cfg['first_dimension']==({'query':'not_reached_allocation_failed'} if outcome=='allocation_error' else {'query':'size0_raises_on_rank_zero'} if outcome=='rank_zero_after_allocation' else {'lower_inclusive':2**31 if outcome=='two31_no_access_return' else 0,'upper_exclusive':2**31+1 if outcome=='two31_no_access_return' else 2**63})
observations={'x_access','y_access','x_missing_bytes','y_missing_bytes','x_encoding','y_encoding','undefined_conversion','output_access','ownership_conflict'}
for pair,outcomes in groups.items():
 r=native_rows[pair]
 if r.get('backend')=='compiled':terminal='returned_or_undefined_conversion'
 elif r.get('backend')=='rejected':terminal='backend_rejected'
 elif r['error_type']=='KeyError':terminal='binder_rejected'
 elif 'not supported in this architecture' in r['error']:terminal='signature_rejected'
 else:terminal='addition_rejected'
 expected={(o,'none') for o in ['allocation_error','rank_zero_after_allocation','context_error',terminal]}
 if terminal=='returned_or_undefined_conversion':expected|={(o,'none') for o in ['compilation_error','pointer_packing_error','grid_limit','launch_setup_error','two31_no_access_return']}|{('native_observation',o) for o in observations}
 assert outcomes==expected
assert len(groups)==484 and len(seen)==len(d['provenance'])==296 and len(d['rules'])==37
for case,c in exact[key].items():
 refs={(r['assumption_id'],r['scope_binding_id']) for r in c['required_binding_refs']};validate_exact_case_links(exact,key,case,refs,links)
 for missing in refs:
  try:validate_exact_case_links(exact,key,case,refs-{missing},links)
  except ValueError:omissions+=1
  else:raise AssertionError('missing port accepted')
assert omissions==296
old=load(P/'evidence/inherited/historical-custom-bounded-v2/receipt.json');catalog=load(P/'evidence/catalog-supplement-task.proposed.json');deps=load(P/'evidence/dependencies-before.json')
assert not set(old['modules'])&set(receipt['modules'])
assert not set(old['modules'])&set(catalog['source_sha256'])
assert not set(old['modules'])&set(deps['project'])
assert catalog['declarations']==[decl]
for p,hv in catalog['source_sha256'].items():assert hf(P/'after'/p if p in receipt['modules'] else R/p)==hv
result={'status':'passed','scope_cases':4498,'dtype_pairs':484,'numeric_ports':296,'numeric_families':37,'omitted_port_mutations_rejected':omissions,'zero_numeric_two31_cases':sum(c['configuration']['outcome']=='two31_no_access_return' for c in cases),'public_statement_sha256':ref['statement_sha256'],'historical_modules_not_in_active_sources_dependencies_catalog_or_authority':True,'historical_module_count':len(old['modules']),'current_module_count':len(receipt['modules']),'GPU_launches':0}
(W/'case-and-port-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
