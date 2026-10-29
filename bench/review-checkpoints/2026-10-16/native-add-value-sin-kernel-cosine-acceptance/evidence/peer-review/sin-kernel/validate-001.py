from pathlib import Path
import collections,hashlib,json,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/sin-kernel-finite-acceptance-wip');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from exact_scope_gate import validate_exact_case_links
from check_original_program_scope import validate_case,validate_deferrals,declaration_scope
load=lambda p:json.loads(Path(p).read_text());hf=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=load(P/'evidence/numerical-authority-delta.json');t=d['requirements'][0];cases=t['cases'];plain=load(P/'evidence/scope-cases.json');row=load(P/'evidence/scope-row-proposal.json');decl=load(P/'evidence/declaration-proposal.json');r=load(P/'evidence/receipt.json')
ref=t['declaration'];actual=declarations(P/'after'/ref['file'])[ref['declaration']];assert actual['kind']=='specification' and actual['statement_sha256']==ref['statement_sha256']
key=(ref['file'],ref['declaration']);exact={key:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in cases}}
assert len(exact[key])==len(cases)==len(plain)==52 and set(exact[key])=={canonical(c) for c in plain}
assert sha(canonical(sorted(plain,key=canonical)))==t['supported_cases_sha256']
for c in plain:validate_case(c)
declaration_scope(decl,row['source_inventory']);validate_deferrals('sin_kernel',row['deferred_numerical_assumptions'],{'rules':d['rules']})
def dtype(x):return {'float8_e8m0fnu':'fp8e8m0fnu','float4_e2m1fn_x2':'packed_fp4e2m1','float16':'fp16','float32':'fp32','float64':'fp64','bfloat16':'bf16','float8_e4m3fn':'fp8e4m3fn','float8_e5m2':'fp8e5m2','float8_e4m3fnuz':'fp8e4m3fnuz','float8_e5m2fnuz':'fp8e5m2fnuz'}.get(x,x)
native=P/load(P/'evidence/native-latest.json')['receipt'];rows={dtype(x['dtype']):x for x in load(native.with_name('records.json'))}
groups=collections.defaultdict(set);links={};seen=set();omissions=0
for c in cases:
 cfg=c['configuration'];name=c['dtype_profile']['inputs']['x'];outcome=cfg['outcome'];groups[name].add(outcome);refs=c['required_binding_refs']
 assert cfg['entrypoint']=='call_kernel' and cfg['BLOCK_SIZE']==1024 and cfg['schedule']=='every_complete_program_permutation'
 expected=({'equals':0} if outcome=='returned_empty' else {'equals':2**31} if outcome=='returned_all_masked' else {'lower_exclusive':(2**31-1)*1024} if outcome=='native_grid_parse_rejected' else {'lower_inclusive':0,'upper_exclusive':2**63})
 assert cfg['n_elements']==expected and len(refs)==(1 if outcome=='returned_defined' else 0)
 assert c['required_callbacks']==sorted({q['callback'] for q in refs})
 for q in refs:
  bits={'fp32':32,'fp64':64}[name];pair=(q['assumption_id'],q['scope_binding_id']);seen.add(pair);links.setdefault(q['callback'],set()).add(pair);port=d['provenance']['::'.join(pair)]
  assert q['callback']==port['callback_declaration']=='SinKernelNative.Numerics.sin'+str(bits)
  assert port['port']=={'name':'b.sin'+str(bits),'ordinal':0,'operation':'sin','input_types':[name],'output_type':name,'scalar_configuration':{'precision':name,'narrowing':False,'load_store_dtype':name,'target':cfg['target']}}
  assert port['configuration']==cfg and port['public_statement']==ref and port['lean_source_sha256']==r['modules']
  assert port['status']=='explicitly_deferred_unvalidated' and port['validation_evidence'] is None and not port['native_execution']
  assert hf(R/port['original_source']['path'])==port['original_source']['sha256']
  for field in ['scalar_source','execution_source']:assert hf(P/'after'/port[field]['path'])==port[field]['sha256']
  assert hf(P/port['raw_ttir']['path'])==port['raw_ttir']['sha256']
  for nr in port['native_receipts']:assert hf(P/nr['path'])==nr['sha256']
for name,outcomes in groups.items():
 row=rows[name];admitted=row['status']=='compiled';terminal='architecture_rejected' if 'not supported in this architecture' in row.get('error','') else row['phase']+'_rejected'
 expected={'allocation_or_setup_failure','native_grid_parse_rejected','returned_empty','returned_all_masked','source_applicability_observation','returned_defined'} if admitted else {'allocation_or_setup_failure',terminal}
 assert outcomes==expected,(name,outcomes,expected)
assert len(groups)==22 and len(seen)==len(d['provenance'])==len(d['rules'])==2
for ck,c in exact[key].items():
 refs={(q['assumption_id'],q['scope_binding_id']) for q in c['required_binding_refs']};validate_exact_case_links(exact,key,ck,refs,links)
 for missing in refs:
  try:validate_exact_case_links(exact,key,ck,refs-{missing},links)
  except ValueError:omissions+=1
  else:raise AssertionError('missing numeric port accepted')
assert omissions==2
retained=load(P/'evidence/retained-proof-review.json')
for path,h in retained['retained_original_files_unchanged'].items():assert hf(R/path)==h
report={'status':'passed','scope_cases':52,'dtype_profiles':22,'admitted_profiles':2,'numeric_ports':2,'numeric_families':2,'omitted_port_mutations_rejected':2,'public_statement_sha256':ref['statement_sha256'],'current_module_count':8,'retained_original_files_unchanged':len(retained['retained_original_files_unchanged']),'GPU_launches':0}
(W/'case-and-port-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
