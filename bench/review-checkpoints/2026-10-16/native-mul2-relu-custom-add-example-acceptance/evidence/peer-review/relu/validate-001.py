from pathlib import Path
import sys,hashlib,json,collections,copy
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;P=Path('/tmp/veritile-standardization-20261015/relu-finite-acceptance-revision1-wip');R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,declarations,sha
from exact_scope_gate import validate_exact_case_links
from check_original_program_scope import validate_case,validate_deferrals,declaration_scope
load=lambda p:json.loads(p.read_text())
d=load(P/'evidence/numerical-authority-delta.json');table=d['requirements'][0];cases=table['cases'];plain=load(P/'evidence/scope-cases.json');row=load(P/'evidence/scope-row-proposal.json');decl=load(P/'evidence/declaration-proposal.json')
ref=table['declaration'];actual=declarations(P/ref['file'])[ref['declaration']];assert actual['kind']=='specification' and actual['statement_sha256']==ref['statement_sha256']
key=(ref['file'],ref['declaration']);exact={key:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in cases}}
assert len(exact[key])==len(cases)==len(plain)
assert set(exact[key])=={canonical(c) for c in plain}
assert sha(canonical(sorted(plain,key=canonical)))==table['supported_cases_sha256']
for c in plain:validate_case(c)
declaration_scope(decl,row['source_inventory']);validate_deferrals('relu_triton_kernel',row['deferred_numerical_assumptions'],{'rules':d['rules']})
links={};numeric=0;omissions=0;seen=set();ranges={};terminals=collections.defaultdict(set);perdtype=collections.Counter()
portmap={'int8':['SINT32'],'int16':['SINT32'],'int32':['SINT32'],'int64':['SINT32'],'uint8':['UINT32'],'uint16':['UINT32'],'uint32':['UINT32'],'uint64':['UINT32'],'bool':['UINT32'],'fp16':['CMP16','EXT16'],'bf16':['EXTBF16','CMP32','EXTBF16'],'fp32':['CMP32'],'fp64':['CMP64','NARROW64']}
for c in cases:
 cfg=c['configuration'];dt=c['dtype_profile']['inputs']['x'];refs=c['required_binding_refs'];size=cfg['size_class'];lo=cfg['N']['lower_inclusive'];hi=cfg['N']['upper_exclusive'];outcome=cfg['outcome']
 assert cfg['block_size']==1024 and cfg['source_variant']=='original' and cfg['entrypoint']=='relu'
 ranges[size]=(lo,hi);terminals[(dt,size)].add(outcome);perdtype[dt]+=1
 expected=portmap[dt] if outcome=='returned_defined' else []
 assert collections.Counter(r['assumption_id'].removeprefix('RELU-ACTUAL-') for r in refs)==collections.Counter(expected),(dt,cfg)
 assert c['required_callbacks']==sorted({r['callback'] for r in refs})
 if refs:numeric+=1
 for rr in refs:
  pair=(rr['assumption_id'],rr['scope_binding_id']);seen.add(pair);links.setdefault(rr['callback'],set()).add(pair)
  p=d['provenance']['::'.join(pair)]
  assert p['configuration']==cfg and p['callback_declaration']==rr['callback']
  assert p['status']=='explicitly_deferred_unvalidated' and p['validation_evidence'] is None and p['native_execution'] is False
  raw=p['raw_ttir'];assert hashlib.sha256((P/raw['path']).read_bytes()).hexdigest()==raw['sha256']
assert len(perdtype)==22 and len(ranges)==8 and len(terminals)==176
ordered=sorted(ranges.values());assert ordered[0][0]==0 and ordered[-1][1]=='unbounded'
for a,b in zip(ordered,ordered[1:]):assert a[1]==b[0]
bound=(2**31-1)*1024
for (dt,size),outcomes in terminals.items():
 lo,hi=ranges[size]
 if dt in ['complex32','complex64','complex128','fp8e8m0fnu','packed_fp4e2m1']:expected='missing_signature_map'
 elif dt in ['fp8e4m3fnuz','fp8e5m2fnuz']:expected='unsupported_signature'
 elif lo>=2**64:expected='nonrepresentable_constexpr'
 elif dt in ['fp8e4m3fn','fp8e5m2']:expected='ordered_comparison_rejected'
 elif lo>bound:expected='grid_parse_rejected'
 elif lo==0:expected='returned_empty'
 else:expected='returned_defined'
 assert outcomes=={'host_or_native_preparation_failure',expected}|({'source_access_outside_model'} if expected=='returned_defined' else set()),(dt,size,outcomes)
assert len(seen)==len(d['provenance'])==len(row['deferred_numerical_assumptions'])==68
for case,c in exact[key].items():
 refs={(r['assumption_id'],r['scope_binding_id']) for r in c['required_binding_refs']}
 validate_exact_case_links(exact,key,case,refs,links)
 for missing in refs:
  try:validate_exact_case_links(exact,key,case,refs-{missing},links)
  except ValueError:omissions+=1
  else:raise AssertionError('missing numerical port accepted')
assert omissions==68
result={'status':'passed','cases_checked':len(cases),'dtype_size_partitions':len(terminals),'dtypes':len(perdtype),'size_partitions':ranges,'numeric_cases':numeric,'exact_ports':len(seen),'numeric_rule_families':len(d['rules']),'omitted_port_mutations_rejected':omissions,'public_statement_sha256':actual['statement_sha256'],'outcomes':dict(collections.Counter(c['configuration']['outcome'] for c in cases)),'gpu_executions':0}
(W/'case-and-port-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
