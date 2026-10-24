"""Independent source-case and exact numeric binding review; no new dtype probes."""
from pathlib import Path
import collections,hashlib,json,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;P=W.parent/'cosine-native-finite-acceptance-revision1-wip';R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from exact_scope_gate import validate_exact_case_links
from check_original_program_scope import validate_case,validate_deferrals,declaration_scope
load=lambda p:json.loads(Path(p).read_text());hf=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert hf(P/'evidence/receipt.json')==(W/'producer-pin.txt').read_text().strip()
d=load(P/'evidence/numerical-authority-delta.json');table=d['requirements'][0];cases=table['cases'];plain=load(P/'evidence/scope-cases.json');row=load(P/'evidence/scope-row-proposal.json');decl=load(P/'evidence/declaration-proposal.json');receipt=load(P/'evidence/receipt.json')
ref=table['declaration'];actual=declarations(P/'after'/ref['file'])[ref['declaration']];assert actual['kind']=='specification' and actual['statement_sha256']==ref['statement_sha256']
key=(ref['file'],ref['declaration']);exact={key:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in cases}}
assert len(exact[key])==len(cases)==len(plain)==268 and set(exact[key])=={canonical(c) for c in plain}
assert sha(canonical(sorted(plain,key=canonical)))==table['supported_cases_sha256']
for c in plain:validate_case(c)
declaration_scope(decl,row['source_inventory']);validate_deferrals('cosine_compute',row['deferred_numerical_assumptions'],{'rules':d['rules']})
host={'CosineNative.HostMath.integer64','CosineNative.HostMath.sqrtSignificand'}
ports=set();links={};host_cases=0;allocation_cases=0;host_ports=0;kernel_ports=0;groups=collections.defaultdict(set);mainlines={}
for case in cases:
 cfg=case['configuration'];name=case['dtype_profile']['inputs']['A'];outcome=cfg['outcome'];refs=case['required_binding_refs'];callbacks={x['callback'] for x in refs};groups[name].add(outcome)
 assert case['required_callbacks']==sorted(callbacks)
 required_host=host & callbacks
 if outcome=='allocation_error':assert not refs;allocation_cases+=1
 else:
  assert required_host==host,(name,outcome);host_cases+=1
  assert cfg['host_numerics']['floating_environment']=='pinned_current_FE_TONEAREST; changes_require_new_native_correspondence'
  assert cfg['host_numerics']['sample_status']=='nonexhaustive_not_a_refinement_proof'
 assert callbacks-host==set() if outcome!='returned_or_undefined_conversion' else True
 if outcome=='returned_or_undefined_conversion' and name in ['fp32','bf16']:
  expected={'CosineNative.Numerics.cos32'} if name=='fp32' else {'CosineNative.Numerics.bfloatTo32','CosineNative.Numerics.cos32','CosineNative.Numerics.toBFloat'}
  assert callbacks-host==expected;mainlines[name]={'callbacks':sorted(callbacks),'case':{k:case[k] for k in ['dtype_profile','configuration']}}
 for port in refs:
  pair=(port['assumption_id'],port['scope_binding_id']);assert pair not in ports;ports.add(pair);links.setdefault(port['callback'],set()).add(pair);p=d['provenance']['::'.join(pair)]
  assert p['configuration']==cfg and p['public_statement']==ref and p['lean_source_sha256']==receipt['modules']
  assert p['callback_declaration']==port['callback'] and p['status']=='explicitly_deferred_unvalidated' and p['validation_evidence'] is None and not p['native_execution']
  if port['callback'] in host:
   host_ports+=1;assert p['correspondence_kind']=='concrete_host_numeric_port' and p['raw_ttir'] is None
   assert p['scalar_path_predicate']=='allocation_success; host_math_executes_before_any_JIT_or_compilation_attempt'
  else:kernel_ports+=1
  for field in ['original_source','scalar_source']:
   v=p[field];path=P/'after'/v['path'] if v['path'] in receipt['modules'] else R/v['path'];assert hf(path)==v['sha256']
  for field in ['execution_source','host_source']:
   v=p[field];assert hf(P/'after'/v['module'])==v['sha256']
  for field in ['raw_ttir','host_math_probe','producer_scalar_inventory']:
   v=p[field]
   if v is not None:assert hf(P/v['path'])==v['sha256']
  for v in p['native_receipts']:assert hf(P/v['path'])==v['sha256']
assert (host_cases,allocation_cases,host_ports,kernel_ports)==(246,22,492,42)
assert len(ports)==len(d['provenance'])==534 and len(d['rules'])==17 and set(mainlines)=={'fp32','bf16'}
omissions=0
for ck,c in exact[key].items():
 refs={(q['assumption_id'],q['scope_binding_id']) for q in c['required_binding_refs']};validate_exact_case_links(exact,key,ck,refs,links)
 for missing in refs:
  try:validate_exact_case_links(exact,key,ck,refs-{missing},links)
  except ValueError:omissions+=1
  else:raise AssertionError('omitted exact numerical port accepted')
assert omissions==534
report={'status':'passed','scope_cases':268,'existing_dtype_evidence_profiles':22,'host_postallocation_cases':246,'allocation_failure_cases_without_host_ports':22,'host_numeric_bindings':492,'kernel_numeric_bindings':42,'numeric_families':17,'omitted_binding_mutations_rejected':534,'public_statement_sha256':ref['statement_sha256'],'mainlines':mainlines,'source_policy':'Existing recorded rows checked; no new dtype matrix or native replay beyond FP32/BF16','host_native_conformance':'explicitly deferred; finite samples are not an all-domain proof','new_three_line_acceptance':False,'acceptance_increment':0}
(W/'case-and-port-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='mainlines'}))
