from pathlib import Path
import copy,json,shutil,subprocess,sys
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015')
NEWREL='2026-10-16/native-vector-quant-transpose-acceptance'
OLDREL='2026-10-16/quant-global-finite-source-acceptance'
WITHDRAWNREL='2026-10-16/quant-transpose-finite-source-acceptance'
B=R/'bench/review-checkpoints'/NEWREL;OLD=R/'bench/review-checkpoints'/OLDREL
WITHDRAWN=R/'bench/review-checkpoints'/WITHDRAWNREL
BASE=R/'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
sys.path.insert(0,str(R/'bench'))
from adapter_common import canonical,hash_file,load,sha,write_json
from current_value_metadata import value_scope
def git(*args):return subprocess.check_output(['git',*args],cwd=R)
def hashes(p):return {q.relative_to(p).as_posix():hash_file(q) for q in p.rglob('*') if q.is_file()}
def exact(p,h):assert hash_file(Path(p))==h,str(p)
def archive(p):
 if p.endswith('.lean'):return p+'.txt'
 if p.endswith(('.olean','.olean.private','.olean.server','.ilean','.ir')):return p+'.bin'
 return p
def prose(value):return value if isinstance(value,str) else '\n'.join(k+': '+v for k,v in value.items())
assert not B.exists() and not git('diff','--cached','--name-only')
base=git('rev-parse','HEAD').decode().strip();assert base=='09eb0985c6bcf794cb2f1aca361982b887c98b32'
assert git('branch','--show-current').decode().strip()=='fp-examples'
exact(WITHDRAWN/'withdrawal.json','5b6c204574ba4b3d6b280378b48cc1c2126a415566ad2ffa055bc0a345fb9f0c')
specs=[
 {'task':'quant_transpose_kernel','tag':'quant-transpose','producer':'quant-transpose-finite-acceptance-revision2-wip','root':'quant-transpose-revision2-root-validation','receipt':'883f7e8e086acb7b20ac7a3f20ba16dfdf697b425ef123d0af7509c877bf188a','review':'54c7e23fb5975ec8fddd7f7bb19304616f9895607fc9f4af313a60e29aacf918','families':16,'ports':140,'cases':1144},
 {'task':'vector_addition','tag':'vector','producer':'vector-finite-acceptance-revision2-wip','root':'vector-native-revision2-root-validation','receipt':'8679242cdbdbe9d8972fe59c011fc89f51ba723de0b233bbe8af07ec32312cc4','review':'2d5529a7a7114a7e846026d44f3d1b8a7b26455294580b7e9a48326074c3f362','families':37,'ports':296,'cases':4498}]
supp=copy.deepcopy(load(OLD/'catalog-supplement.json'))
prior=json.loads(git('show',base+':bench/original_program_scope.json'))
current=load(R/'bench/original_program_scope.json')
prior_rows={t['task']:t for t in prior['tasks']};current_rows={t['task']:t for t in current['tasks']}
assert all(t==prior_rows[t['task']] for t in current['tasks'] if t['task']!='quant_transpose_kernel')
sources={};inputs={};semantics={};decls=[];packages={};value_hashes={}
for item in specs:
 P=T/item['producer'];V=T/item['root'];exact(P/'evidence/receipt.json',item['receipt']);exact(V/'review.json',item['review'])
 r=load(P/'evidence/receipt.json');review=load(V/'review.json');rr=load(V/'review-receipt.json')
 assert rr['review_sha256']==item['review'] and review['reviewed_candidate_source_sha256']==r['modules']
 for p,h in r['artifacts'].items():exact(P/p,h)
 for p,h in rr['artifacts'].items():exact(V/p,h)
 item['module_count']=len(r['modules']);packages[item['task']]=(P,V,r,review,rr)
 for p,h in r['modules'].items():
  src=P/'after'/p if item['tag']=='vector' else P/p;exact(src,h)
  if (R/p).exists():
   assert item['tag']=='quant-transpose';exact(R/p,load(T/'quant-transpose-finite-acceptance-revision1-wip/evidence/receipt.json')['modules'][p])
  sources[p]=h
 task=next(t for t in supp['tasks'] if t['task']==item['task'])
 decl=load(P/'evidence/declaration-proposal.json');semantics[item['task']]=prose(review['semantic_review']);decl['review_basis']=semantics[item['task']]
 task['declarations'].append(decl);task['source_sha256'].update(r['modules'])
 task['required_value_stages'] += [s for s in decl['covers_stages'] if s not in task['required_value_stages']]
 value_hashes[item['task']]=sha(canonical(value_scope(task)));decls.append(decl)
 oldrow=prior_rows[item['task']];inputs.update(oldrow['input_sha256']['lean']);inputs.update(oldrow['input_sha256']['python']);inputs.update(r['modules'])
 deps=load(V/'dependencies-before-root.json')
 for key,v in deps['project'].items():
  p=Path(v['source']).relative_to(R).as_posix() if 'source' in v else key
  inputs[p]=v['sha256']
state={'base_commit':base,'predecessor_file_sha256':hashes(OLD),'withdrawn_checkpoint_sha256':hashes(WITHDRAWN),'metadata_baseline_sha256':hashes(BASE),'new_source_sha256':sources,'packages':specs,'prior_catalog_publics':955,'prior_supplement_declarations':62,'prior_numeric_rules':100,'prior_source_scope':{'verified':1,'verified_with_deferred_numerics':1,'pending':181},'dirty_before':git('status','--porcelain').decode(),'previous_working_ledger_sha256':hash_file(R/'bench/original_program_scope.json')}
B.mkdir(parents=True);(B/'evidence').mkdir();retained={};relocations={}
def retain(src,rel,h=None):
 src=Path(src);h=h or hash_file(src);exact(src,h);dest=B/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);retained[rel]=h
retain(R/'bench/original_program_scope.json','evidence/inherited-withdrawal-ledger.json')
retain(R/'bench/original_program_scope.md','evidence/inherited-withdrawal-progress.md')
for item in specs:
 P,V,r,review,rr=packages[item['task']];prefix='evidence/producer/'+item['tag']+'/'
 for p,h in r['modules'].items():
  src=P/'after'/p if item['tag']=='vector' else P/p
  dest=R/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
  retain(src,prefix+r['checkpoint_module_source_paths'][p],h)
 for p,h in {**r['artifacts'],'evidence/receipt.json':item['receipt']}.items():
  mapped=archive(p);retain(P/p,prefix+mapped,h)
  if mapped!=p:relocations[prefix+p]=prefix+mapped
 for p,h in rr['artifacts'].items():retain(V/p,'evidence/root-validation/'+item['tag']+'/'+archive(p),h)
 retain(V/'review-receipt.json','evidence/root-validation/'+item['tag']+'/review-receipt.json')
 retain(V/'review.json','evidence/root-validation/'+item['tag']+'/review.json',item['review'])
rootreview={'schema':1,'kind':'native_vector_quant_transpose_source_integration_review','reviewer':'root','status':'reviewed_for_finite_scope_acceptance_with_registered_numerical_deferrals','base_commit':base,'reviewed_input_sha256':inputs,'reviewed_value_scope_sha256':value_hashes,'reviewed_metadata_corrections_sha256':{},'semantic_review':semantics,'metadata_baseline':load(OLD/'root-review.json')['metadata_baseline'],'new_public_contracts':decls,'packages':specs,'native_execution':False,'acceptance_boundary_policy_sha256':hash_file(R/'bench/original_program_acceptance.md')}
write_json(B/'root-review.json',rootreview)
for item in specs:
 task=next(t for t in supp['tasks'] if t['task']==item['task'])
 task['review']={'path':NEWREL+'/root-review.json','sha256':hash_file(B/'root-review.json'),'source_hashes_pointer':['reviewed_input_sha256']}
write_json(B/'catalog-supplement.json',supp);assert sum(len(t['declarations']) for t in supp['tasks'])==64
write_json(B/'evidence/preparation-state.json',state)
retain(Path(__file__),'evidence/prepare.py')
write_json(B/'evidence-index.json',{'schema':1,'retained_exact_artifacts':retained,'new_source_sha256':sources,'archived_producer_path_mapping':relocations,'note':'All frozen receipt bytes are unchanged. Historical Lean sources use .txt; generated Lean objects explicitly bound as evidence use .bin. These files cannot be discovered as active modules.'})
f=R/'.github/workflows/bench-audit.yml';s=f.read_text();assert s.count(WITHDRAWNREL)==3;f.write_text(s.replace(WITHDRAWNREL,NEWREL))
(B/'README.md').write_text('''# Native vector and quantize transpose source acceptance

This batch adds the independently reviewed revision2 native vector contract and
replaces the uncommitted quantize-transpose proposal with its reviewed revision2.
Both cover the original finite host extent domain and actual typed integer masks,
wrapped addresses, execution order, returned metadata and independent byte results.
Signed grid parsing precedes pointer extraction. Earlier bounded declarations,
numerical registrations, failed attempts and the withdrawn quantize-transpose
checkpoint remain preserved.

The source acceptance proposal covers vector_addition and quant_transpose_kernel.
It adds 53 numerical families, 436 exact ports and 5,642 cases; these numerical
relations remain deferred. No GPU or official comparator success is claimed.
The old dimension-free quantize-transpose admission is historical evidence only.

Frozen producer and root review receipts retain exact bytes. Historical Lean
sources are archived as .lean.txt and receipt-bound generated objects as .bin,
outside active source discovery. The quantize-transpose producer omitted a
historical direct Range object hash; root captured it before independent rebuild.
Vector's producer named its external imports without historical object hashes;
root separately captured them before rebuild. Neither gap is rewritten as a
historical capture. Integration validation is recorded separately below.
''')
print(json.dumps({'checkpoint':str(B),'modules':len(sources),'publics':2,'retained_files':len(retained),'archive_relocations':len(relocations),'root_review_sha256':hash_file(B/'root-review.json')},indent=2))
