from pathlib import Path
import collections,hashlib,json,re,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/add-value-finite-acceptance-revision1-wip');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from exact_scope_gate import validate_exact_case_links
from check_original_program_scope import validate_case,validate_deferrals,declaration_scope
load=lambda p:json.loads(Path(p).read_text());hf=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=load(P/'evidence/numerical-authority-delta.json');table=d['requirements'][0];cases=table['cases'];plain=load(P/'evidence/scope-cases.json');row=load(P/'evidence/scope-row-proposal.json');decl=load(P/'evidence/declaration-proposal.json');receipt=load(P/'evidence/receipt.json')
ref=table['declaration'];actual=declarations(P/'after'/ref['file'])[ref['declaration']]
assert actual['kind']=='specification' and actual['statement_sha256']==ref['statement_sha256']
key=(ref['file'],ref['declaration']);exact={key:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in cases}}
assert len(exact[key])==len(cases)==len(plain)==238
assert set(exact[key])=={canonical(c) for c in plain}
assert sha(canonical(sorted(plain,key=canonical)))==table['supported_cases_sha256']
for c in plain:validate_case(c)
declaration_scope(decl,row['source_inventory']);validate_deferrals('add_value',row['deferred_numerical_assumptions'],{'rules':d['rules']})
def dtype(name):return {'float16':'fp16','bfloat16':'bf16','float32':'fp32','float64':'fp64','float8_e4m3fn':'fp8e4m3fn','float8_e5m2':'fp8e5m2','float8_e4m3fnuz':'fp8e4m3fnuz','float8_e5m2fnuz':'fp8e5m2fnuz','float8_e8m0fnu':'fp8e8m0fnu','float4_e2m1fn_x2':'packed_fp4e2m1'}.get(name,name)
profiles={dtype(p['dtype']):p for p in load(P/'evidence/scalar-operation-inventory.json')['profiles']}
native=P/load(P/'evidence/native-latest.json')['receipt'];native_rows={dtype(r['dtype']):r['result'] for r in load(native)['records']}
groups=collections.defaultdict(set);links={};seen=set();omissions=0
for c in cases:
 cfg=c['configuration'];name=c['dtype_profile']['inputs']['x'];refs=c['required_binding_refs'];outcome=cfg['outcome'];groups[name].add((outcome,cfg['native_observation']))
 assert cfg['BLOCK_SIZE']==1024 and cfg['entrypoint']=='puzzle1'
 assert cfg['value']=={'python':10,'runtime_type':'fp32','bits':0x41200000,'constexpr':False}
 assert cfg['numel']==({'query':'not_reached_allocation_or_cuda_failure'} if outcome in ['allocation_error','cuda_assertion'] else {'lower_inclusive':2**31 if outcome=='two31_no_access_return' else 0,'upper_exclusive':2**31+1 if outcome=='two31_no_access_return' else 2**63})
 selected=profiles[name]['operations'] if outcome=='returned_or_undefined_conversion' else []
 assert len(refs)==len(selected) and c['required_callbacks']==sorted({rr['callback'] for rr in refs})
 for ordinal,(rr,op) in enumerate(zip(refs,selected)):
  pair=(rr['assumption_id'],rr['scope_binding_id']);seen.add(pair);links.setdefault(rr['callback'],set()).add(pair);port=d['provenance']['::'.join(pair)]
  assert port['port']['name']==op['port'] and port['port']['ordinal']==ordinal and port['port']['operation']==op['operation'] and port['port']['scalar_configuration']==op['configuration']
  cb='VeriTile.Triton.'+op['port'] if op['port'].startswith('FP.') else 'VeriTile.Bench.TritonBenchG.VectorAddition.FP64TypedCore.Numerics.'+op['port'].split('.')[-1] if op['port'].startswith('wide.') else 'AddValueNative.Numerics.'+op['port']
  assert rr['callback']==port['callback_declaration']==cb
  assert port['configuration']==cfg and port['public_statement']==ref and port['lean_source_sha256']==receipt['modules']
  assert port['status']=='explicitly_deferred_unvalidated' and port['validation_evidence'] is None and not port['native_execution']
  for field in ['raw_ttir','frontend_receipt','producer_scalar_inventory']:assert hf(P/port[field]['path'])==port[field]['sha256']
  for field in ['original_source','scalar_source']:
   path=port[field]['path'];assert hf(P/'after'/path if path in receipt['modules'] else R/path)==port[field]['sha256']
  for field in ['execution_source','host_source']:assert hf(P/'after'/port[field]['module'])==port[field]['sha256']
for name,outcomes in groups.items():
 n=native_rows[name];admitted=n['status']=='compiled';terminal='returned_or_undefined_conversion' if admitted else 'architecture_rejected' if 'not supported in this architecture' in n['exception'] else 'binder_rejected'
 expected={(v,'none') for v in ['allocation_error','cuda_assertion','context_error',terminal]}
 if admitted:
  expected|={(v,'none') for v in ['compilation_error','grid_limit','pointer_packing_error','launch_setup_error','two31_no_access_return']}
  expected|={('native_observation',v) for v in ['input_access','missing_bytes','undefined_conversion','output_access','ownership_conflict']}
 assert outcomes==expected,(name,outcomes,expected)
assert len(groups)==22 and len(profiles)==15 and len(seen)==len(d['provenance'])==41 and len(d['rules'])==15
for case,c in exact[key].items():
 refs={(rr['assumption_id'],rr['scope_binding_id']) for rr in c['required_binding_refs']};validate_exact_case_links(exact,key,case,refs,links)
 for missing in refs:
  try:validate_exact_case_links(exact,key,case,refs-{missing},links)
  except ValueError:omissions+=1
  else:raise AssertionError('missing port accepted')
assert omissions==41
result={'status':'passed','scope_cases':238,'dtypes':22,'admitted_profiles':15,'numeric_ports':41,'numeric_families':15,'omitted_port_mutations_rejected':omissions,'public_statement_sha256':ref['statement_sha256'],'current_module_count':len(receipt['modules']),'GPU_launches':0}
(W/'case-and-port-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
