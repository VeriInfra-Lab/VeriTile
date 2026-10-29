from pathlib import Path
from datetime import datetime,timezone
import json,sys,copy,importlib.util,re
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'quant-transpose-finite-acceptance-revision1-wip';V=T/'quant-transpose-revision1-root-validation';PEER=T/'quant-transpose-revision1-matrix-review'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json,declarations,imports
from check_original_program_scope import declaration_scope,validate_case,validate_deferrals
from exact_scope_gate import validate_exact_case_links
checked={}
def exact(p,h=None):
 p=Path(p);actual=hash_file(p)
 if h is not None:assert actual==h,str(p)
 checked[str(p)]=actual;return actual
rh='63392bac5342812d467c2bc499e7870e2609d49fa74faaf01731c1391b1dcf83';ph='57aab0306494ad08cc9f4d29a404ae2bbd5bf182cba5c9f9f660dd861f6e1354'
exact(P/'evidence/receipt.json',rh);exact(PEER/'review-receipt.json',ph);r=load(P/'evidence/receipt.json');peer=load(PEER/'review-receipt.json');assert r['frozen'] and peer['frozen'] and r['revision']==1
for p,h in r['artifacts'].items():exact(P/p,h)
for p,h in r['modules'].items():exact(P/p,h)
for p,h in r['module_objects'].items():exact(P/p,h)
for p,h in peer['artifacts'].items():exact(PEER/p,h)
exact(PEER/'peer-review.json','adae663044f6566615ed5dd70cc9e23fb476193f6b2d5ba78a23ed8d9c43108e')
project={};external={}
for p,ref in r['dependencies']['captures'].items():
 exact(P/p,ref['sha256']);assert ref['verified_unchanged'];dep=load(P/p)
 for mod,v in dep.get('project',{}).items():
  exact(v['source'],v['sha256']);exact(v['olean'],v['olean_sha256']);project[mod]=v
 for mod,v in dep.get('external_objects',{}).items():exact(v['olean'],v['sha256']);external[mod]=v
 for path,h in dep.get('input_sha256',{}).items():exact(path,h)
assert len(project)==177 and len(external)==33
known=set(project)|set(external)|{p.removesuffix('.lean').replace('/','.') for p in r['modules']}
for p in r['modules']:
 for mod in imports((P/p).read_text()):assert mod in known,(p,mod)
comp=load(PEER/'independent-compiles.json');assert comp['status']=='passed' and comp['receipt_sha256']==rh and len(comp['modules'])==18
axioms=noncircular=0
for c in comp['modules']:
 assert c['exit_code']==0 and c['source_unchanged'] and c['source_sha256']==r['modules'][c['module']]
 exact(PEER/'overlay'/c['module'],c['source_sha256']);exact((PEER/'overlay'/c['module']).with_suffix('.olean'),c['object_sha256']);exact(PEER/c['log'],c['log_sha256'])
 log=(PEER/c['log']).read_text();assert not re.search(r'error:|sorryAx|DISALLOWED',log)
 axioms+=log.count('axiom footprint ⊆ standard base');noncircular+=log.count('— non-circular ✓')
assert (axioms,noncircular)==(68,14)
public=[]
for p in r['modules']:
 for name,d in declarations(P/p).items():
  if d['kind']=='specification':public.append({'file':p,'declaration':name,'kind':d['kind'],'statement_sha256':d['statement_sha256']})
assert len(public)==1 and public[0]['statement_sha256']=='e74f9aed7ed2a1bd056dbb911f0a92480007153689ce375597019cb71c7ac2ba'
delta=load(P/'evidence/numerical-authority-delta.json');row=load(P/'evidence/scope-row-proposal.json');decl=load(P/'evidence/declaration-proposal.json')
current=next(t for t in load(R/'bench/original_program_scope.json')['tasks'] if t['task']=='quant_transpose_kernel');assert row['source_inventory']==current['source_inventory'];assert row['structural_gaps']==[]
declaration_scope(decl,row['source_inventory']);validate_deferrals('quant_transpose_kernel',row['deferred_numerical_assumptions'],{'rules':delta['rules']})
assert len(delta['rules'])==16 and len(delta['provenance'])==140 and len(delta['requirements'])==1
table=delta['requirements'][0];cases=load(P/'evidence/scope-cases.json');assert len(cases)==len(table['cases'])==863
assert table['supported_cases_sha256']==sha(canonical(sorted(cases,key=canonical)))
for case in cases:validate_case(case)
assert {canonical(c) for c in cases}=={canonical({k:c[k] for k in ['dtype_profile','configuration']}) for c in table['cases']}
key=(table['declaration']['file'],table['declaration']['declaration']);expected={key:{canonical({k:c[k] for k in ['dtype_profile','configuration']}):c for c in table['cases']}};links={}
for c in table['cases']:
 for ref in c['required_binding_refs']:links.setdefault(ref['callback'],set()).add((ref['assumption_id'],ref['scope_binding_id']))
mutations=0
for ck,c in expected[key].items():
 refs={(ref['assumption_id'],ref['scope_binding_id']) for ref in c['required_binding_refs']};validate_exact_case_links(expected,key,ck,refs,links)
 for omit in refs:
  try:validate_exact_case_links(expected,key,ck,refs-{omit},links)
  except ValueError:mutations+=1
  else:raise AssertionError('omitted numerical port accepted')
assert mutations==140
spec=importlib.util.spec_from_file_location('qt_review_adapter',P/'authority_adapter.py');ad=importlib.util.module_from_spec(spec);spec.loader.exec_module(ad)
for evidence in delta['native_evidence']:
 assert ad.native_receipt_inputs(P,evidence['receipt'],delta['original_source']['sha256'])==evidence['retained_inputs']
for p,h in ad.project_dependency_inputs(P,delta['dependency_snapshots']).items():exact(R/p,h)
for port in delta['provenance'].values():
 assert port['lean_source_sha256']==r['modules'] and port['original_source']==delta['original_source']
 exact(R/port['legacy_route_source']['path'],port['legacy_route_source']['sha256'])
exact(R/delta['original_source']['path'],delta['original_source']['sha256']);exact(R/'bench/original_program_acceptance.md');exact(Path(__file__))
V.mkdir(exist_ok=True);assert not (V/'review.json').exists()
semantic='Root read the original Python, public acceptance statement, actual integer expression tree, byte reads/writes, native program/grid/host composition, typed scalar expressions, admission, metadata/physical bindings and conditional refinement. All source grouped coordinates retain pid%group_size and each actual constexpr/i32/i64 promotion and intermediate wrap, including singleton large strides and negative active masks. Input and output pointers sign-extend selected offsets and wrap at64bits; active accesses alone are classified. The whole input tile loads before the unmasked inverse and concrete output stores. Host abs/max traverses the logical strided view and its original maximum/inverse are physically written before kernel execution; returned maximum is not substituted with the reciprocal. Independent plans use host-prepared bytes, actual masks/addresses and independent scalar expressions; evolving-memory execution is related by input/inverse-output disjointness. Every launch and finite autotune trial is composed, ownership is reset per launch, output bytes have an independent last-write formula and exterior frame, and returned max/inverse bytes are preserved. The main theorem has no Safe/ArithmeticRefines premise: actual grid-limit, access/ownership/missing-byte/integer-division/conversion observations and successful actual-address returns are partitioned. Observations assert no native trap, rollback or postfailure state. Ordinary signed64 Torch metadata and concrete fresh allocation remain trusted primitives; all22 canonical CPU/CUDA dtype families retain source-first abs/max/rank/stride/signature/llrint/pointer failures and primitive exceptions. Eight admitted scalar storage routes retain exact promotions, nested multiplication, defined nearest-even llrint64 and int8 narrowing, with all16 numerical families/140ports precisely registered. The logical transpose formula is conditional on ArithmeticRefines and does not exclude safe wrapped actual-address executions. Existing fp32 source interpreter and proofs remain connected through the legacy bridge. Root rechecked every producer/peer artifact and dependency, all863 case schemas and all140 omitted-port rejections; the independently rebuilt18 peer modules have68 named clean axiom checks and14 noncircular checks. Current repository build and full authority/catalog/ledger integration remain required before acceptance.'
report={'schema':1,'kind':'quant_transpose_revision1_finite_source_root_review','reviewer':'root','status':'substantively_reviewed_for_finite_scope_integration','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'producer_receipt_sha256':rh,'peer_review_receipt_sha256':ph,'checked_file_sha256':checked,'integration_sources':r['modules'],'public_contracts':public,'semantic_review':semantic,'checked_counts':{'modules':18,'producer_artifacts':len(r['artifacts']),'project_dependencies':177,'external_dependencies':33,'peer_clean_axiom_checks':68,'peer_noncircular_checks':14,'source_cases':863,'numeric_families':16,'exact_ports':140,'root_omitted_port_mutations':140,'producer_attempts_including_inherited':110,'producer_failed_attempts_preserved':19},'proposed_scope_status':'verified_with_deferred_numerics','evidence_limitations':['Peer independent build is verified here; current integration build remains required.','No GPU execution, native numerical validation or official comparator success.','No defined postfailure memory, native exception or race winner claimed for interpreter observations.','Historical revision1 predecessor remains preserved and unaccepted.'],'acceptance_increment_before_integration':0,'numerical_conformance_validated':False,'native_execution':False}
write_json(V/'review.json',report)
print(json.dumps({'status':report['status'],'review_sha256':hash_file(V/'review.json'),'checked_counts':report['checked_counts']},indent=2))
