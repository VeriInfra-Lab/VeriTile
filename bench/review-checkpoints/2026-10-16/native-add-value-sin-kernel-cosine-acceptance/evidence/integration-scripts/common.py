"""Candidate integration helpers. Importing this module never mutates the repo."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import argparse, ast, gzip, hashlib, importlib.util, json, os, re, shutil, subprocess, sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'specs.json').read_text())
EXPECTED = SPEC['expected_after']
LIMIT = 100 * 1024 * 1024
STORAGE_CHECKPOINT = 'bench/review-checkpoints/2026-10-16/numerical-queue-gzip-migration'
QUEUE = 'bench/tritonbench_g/deferred_numerical_experiments.json.gz'
QUEUE_MARKDOWN = 'bench/tritonbench_g/deferred_numerical_experiments.md'
BASELINE = {'accepted':8,'pending':175,'public_declarations':961,'rules':239,'bindings':2366,'cases':22317,'tables':12}
PRIMARY_POLICY = 'bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/primary-version-policy.md'
PRIMARY_POLICY_SHA256 = '10a59ed41d81aee0e5bb82b81a998116564a8dd7c378516a0a4edf894b07be85'
PROTECTED = [
 'bench/adapter_common.py', 'bench/check_original_program_scope.py',
 'bench/current_source_catalog.py', 'bench/current_source_audit.py',
 'bench/current_value_metadata.py', 'bench/exact_numerical_bindings.py',
 'bench/exact_scope_gate.py', 'bench/deferred_numerical_experiments.py',
 'lakefile.toml', 'lake-manifest.json', 'lean-toolchain',
 'bench/original_program_acceptance.md',
 'bench/tests/test_numerical_queue_storage.py', 'bench/tests/test_numerical_queue_migration.py',
 PRIMARY_POLICY,
]
BASE = 'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
SOURCE_MAP = 'bench/current_source_selections.json'
LEGACY = 'bench/tritonbench_g/current_completion_inventory.json'
REVIEW = 'bench/review-checkpoints/2026-10-15/raw-copy-software-metadata/catalog-review-rebased.json'
TARGETS = ['VeriTile','VeriTileMath','VeriTileFull','TritonBenchGVerified',
 'TritonBenchGInProgress','TritonBenchGLegacy','TritonBenchSpecExamples','FPFrontendRegressions']
OLD_ACCEPTED = {'matrix_transpose','quantize_global','vector_addition','quant_transpose_kernel','triton_mul2','relu_triton_kernel','vector_addition_custom','add_example'}
NEW_TASKS = {p['task'] for p in SPEC['packages']}

def require(ok, message):
 if not ok: raise ValueError(message)

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()

def exact(path, expected):
 require(Path(path).is_file() and digest(path)==expected, 'Changed/missing input: '+str(path))

def no_duplicates(pairs):
 result={}
 for k,v in pairs:
  require(k not in result,'Duplicate JSON key: '+k);result[k]=v
 return result

def load(path):
 path=Path(path);raw=path.read_bytes()
 if path.name.endswith('.json.gz'):raw=gzip.decompress(raw)
 return json.loads(raw,object_pairs_hook=no_duplicates)

def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def object_sha(obj):return hashlib.sha256(canonical(obj)).hexdigest()

def write(path, obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 raw=(json.dumps(obj,indent=2,ensure_ascii=False)+'\n').encode()
 if path.name.endswith('.json.gz'):
  import io
  out=io.BytesIO()
  with gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0) as f:f.write(raw)
  raw=out.getvalue()
 path.write_bytes(raw)

def relative(path):
 text=str(path);p=Path(text);require(not p.is_absolute() and '..' not in p.parts and p.as_posix()==text and '\\' not in text and text not in ('','.'),'Unsafe relative path: '+str(p))
 return p

def archive(path):
 text=relative(path).as_posix()
 if text.endswith('.lean'):return text+'.txt'
 if text.endswith(('.olean','.olean.private','.olean.server','.ilean','.ir','.pyc')):return text+'.bin'
 return text

def hashes(root):return {p.relative_to(root).as_posix():digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}
def prose(value):return value if isinstance(value,str) else '\n'.join(k+': '+(v if isinstance(v,str) else json.dumps(v,sort_keys=True)) for k,v in value.items())
def git(root,*args,input=None):return subprocess.check_output(['git',*args],cwd=root,input=input)

def explicit_base(root,expected,clean=False):
 require(isinstance(expected,str) and re.fullmatch(r'[0-9a-f]{40}',expected),'Supply the full actual post-gzip commit; no historical/default HEAD alias')
 require(git(root,'rev-parse','HEAD').decode().strip()==expected,'HEAD differs from explicit post-gzip base')
 require(git(root,'cat-file','-t',expected).decode().strip()=='commit','Base is not a commit')
 if clean:require(not git(root,'status','--porcelain=v1','--untracked-files=all'),'Requires clean committed post-gzip base')

def external_output(root,path):
 out=Path(path).resolve();require(out.is_relative_to(Path('/tmp')) and not out.is_relative_to(Path(root).resolve()),'Run/report output must be outside the repository under /tmp');return out

def verify_scripts(root,expected):
 root=Path(root);exact(root/'candidate-receipt.json',expected);receipt=load(root/'candidate-receipt.json')
 require(receipt.get('frozen') is True and receipt.get('artifacts'),'Unfrozen script package')
 for rel,h in receipt['artifacts'].items():exact(root/relative(rel),h)
 return receipt

def copy_scripts(destination,expected):
 receipt=verify_scripts(HERE,expected)
 for rel,h in {'candidate-receipt.json':expected,**receipt['artifacts']}.items():
  dest=destination/relative(rel);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/rel,dest);exact(dest,h)
 verify_scripts(destination,expected)

def active_gzip(root):
 require((root/QUEUE).is_file(),'Active numerical queue must explicitly be gzip')
 for rel in [QUEUE.removesuffix('.gz'),'bench/original_program_scope.json']:
  require(not (root/rel).exists() and not (root/rel).is_symlink(),'Competing historical active path: '+rel)

def verify_storage_base(root,expected_validation):
 active_gzip(root);cp=root/STORAGE_CHECKPOINT
 exact(cp/'validation/check.json',expected_validation)
 report=load(cp/'validation/check.json');require(report['status']=='passed' and report['queue_bytes_and_all_legacy_rows_preserved'] is True and report['legacy_scope_result_unchanged'] is True,'Storage migration validation incomplete')
 snapshot=report['coherent_snapshot'];exact(root/relative(snapshot['manifest']),snapshot['sha256']);manifest=load(root/snapshot['manifest'])
 exact(cp/'plan.json',manifest['migration_plan_sha256']);plan=load(cp/'plan.json')
 scriptroot=cp/'evidence/integration-scripts';exact(scriptroot/'receipt.json',plan['scripts_receipt_sha256'])
 for rel,h in load(scriptroot/'receipt.json')['artifacts'].items():exact(scriptroot/relative(rel),h)
 for rel,h in manifest['root_input_sha256'].items():exact(root/relative(rel),h)
 for ref in manifest['documents'].values():exact((root/snapshot['manifest']).parent/relative(ref['path']),ref['sha256'])
 require(manifest['legacy_status_counts']=={'verified':1,'verified_with_deferred_numerics':7,'pending':175},'Storage snapshot has wrong legacy baseline')
 exact(root/PRIMARY_POLICY,PRIMARY_POLICY_SHA256)
 for rel,h in load(HERE/'future-protected-pins.json').items():exact(root/relative(rel),h)
 return report

def new_task_record(package,semantics):
 item=package['spec'];proposal=package['producer']/'evidence/catalog-supplement-task.proposed.json'
 if proposal.exists():
  task=deepcopy(load(proposal));require(task['task']==item['task'] and len(task['declarations'])==1,'Unexpected reviewed task proposal');task['declarations']=[];return task
 require(item['task'] in {'sin_kernel','cosine_compute'},'Missing reviewed task proposal')
 name=item['task'];prefix='bench/tritonbench_g/'+name+'/'
 selected={prefix+'Proofs/NativeSource.lean':['lowerBody','compiledSource'],prefix+'Proofs/NativeProgram.lean':['executeProgram','programPlan'],prefix+'Proofs/NativeScalar.lean':['source','result']}
 if name=='cosine_compute':selected[prefix+'Proofs/NativeHostMath.lean']=['prepare','block','grid']
 view={'id':name+'_native_original_source','variant':'original_source_translation','frontend':'task_local_exact_ast','ordinary_shared_dsl_admission':False,'memory_semantics':'sequentially_consistent_schedule','original_python':prefix+name+'.py','candidate_python':None,'selected_definitions':selected,'entry_aliases':{prefix+'NativeCorrect.lean':['executeAllocated','executeHost']},'translation_trust_boundary':semantics}
 return {'task':name,'required_value_stages':[],'source_sha256':{},'source_views':[view],'declarations':[],'review':None}

def primary_inventory(packages):
 rows=[]
 for package in packages:
  item=package['spec'];tables=package['delta']['requirements'];versions={}
  for version,dtype in [('FP32','fp32'),('BF16','bf16')]:
   cases=[c for table in tables for c in table['cases'] if set(c['dtype_profile']['inputs'].values())=={dtype}]
   versions[version]={'proof_status':'pending','support_status':'unreviewed','source_cases':len(cases),'outcomes':dict(Counter(c['configuration'].get('outcome','unspecified') for c in cases)),'source_compute_types':sorted({x for c in cases for x in c['dtype_profile'].get('compute',{}).values()}),'numerical_binding_refs':len({canonical(r) for c in cases for r in c['required_binding_refs']})}
  rows.append({'task':item['task'],'Real':{'proof_status':'pending','support_status':'mathematical_not_runtime','note':'No Real completion is inferred from the finite native source package.'},**versions,'root_review_sha256':item['review'],'root_primary_version_mapping':package['review'].get('primary_version_mapping'),'primary_completion_increment':0})
 return {'schema':1,'kind':'source_case_primary_applicability_inventory_not_acceptance','tasks':rows,'policy':{'path':PRIMARY_POLICY,'sha256':PRIMARY_POLICY_SHA256},'maintained_variants_included':False,'primary_completion_increment':0}

def primary_inventory_markdown(inventory):
 lines=['# Primary-version applicability inventory','','These are source-case counts from the frozen reviewed packages. They establish no primary-version completion. Real remains a mathematical model; FP32/BF16 require separate domain, composition, rejection and numerical review under the approved policy. Source-required scalar/index/host types remain intact. No maintained variant replaces an original source slot.','','| Task | FP32 source cases | BF16 source cases | Primary completion |','| --- | ---: | ---: | --- |']
 for row in inventory['tasks']:lines.append(f"| {row['task']} | {row['FP32']['source_cases']} | {row['BF16']['source_cases']} | Pending for each version |")
 lines+=['','Exact outcome partitions, source compute types, numerical reference counts and root review pins are in `evidence/primary-version-applicability.json`. Masked returns cannot establish functional completion, and host numerical requirements are not erased by a masked/empty kernel. Existing wider proof evidence remains available.','']
 return '\n'.join(lines)

def arguments(description):
 p=argparse.ArgumentParser(description=description)
 p.add_argument('--root',type=Path,default=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile'))
 p.add_argument('--packages',type=Path,default=HERE.parent)
 # These are explicit names, never extension or existence fallbacks.
 p.add_argument('--ledger',default='bench/original_program_scope.json.gz')
 p.add_argument('--baseline-ledger',default='bench/original_program_scope.json.gz')
 p.add_argument('--predecessor-catalog',default='current-source-catalog.json.gz')
 p.add_argument('--predecessor-supplement',default='catalog-supplement.json.gz')
 p.add_argument('--predecessor-review',default='root-review.json.gz')
 p.add_argument('--expected-base-commit')
 p.add_argument('--scripts-receipt-sha256')
 p.add_argument('--storage-validation-sha256')
 return p

def runtime_arguments(args):
 values=[('root',str(args.root)),('packages',str(args.packages)),('ledger',args.ledger),('baseline-ledger',args.baseline_ledger),('predecessor-catalog',args.predecessor_catalog),('predecessor-supplement',args.predecessor_supplement),('predecessor-review',args.predecessor_review),('expected-base-commit',args.expected_base_commit),('scripts-receipt-sha256',args.scripts_receipt_sha256),('storage-validation-sha256',args.storage_validation_sha256)]
 require(all(isinstance(value,str) and value for _,value in values),'Explicit runtime arguments are incomplete')
 return [value for key,item in values for value in ('--'+key,item)]

def context(args):
 relative(args.ledger);relative(args.baseline_ledger)
 for name in [args.predecessor_catalog,args.predecessor_supplement,args.predecessor_review]:relative(name)
 r=args.root.resolve();b=r/'bench/review-checkpoints'/SPEC['intended_checkpoint']
 return r,b,r/'bench/review-checkpoints'/SPEC['predecessor']

def bench_imports(root):
 sys.path.insert(0,str(root/'bench'))
 import adapter_common as ac
 return ac

def package_inputs(args, check_hashes=True):
 """Return exactly receipt-bound inputs; no directory-based evidence guessing."""
 out=[];destinations={}
 for item in SPEC['packages']:
  p=args.packages/item['producer'];v=args.packages/item['root'];w=args.packages/item['peer']
  pinned=[(p/'evidence/receipt.json',item['receipt']), (v/'review.json',item['review']),
   (v/'review-receipt.json',item['root_receipt_sha256']),
   (w/item['peer_receipt'],item['peer_receipt_sha256']), (w/'peer-review.json',item['peer_review_sha256'])]
  for path,h in pinned:exact(path,h)
  receipt=load(p/'evidence/receipt.json');review=load(v/'review.json')
  rr=load(v/'review-receipt.json');pr=load(w/item['peer_receipt'])
  require(receipt['frozen'] is True and rr['frozen'] is True and pr['frozen'] is True,'Unfrozen package')
  require(review['reviewer']=='root' and review['status']=='substantively_reviewed_for_finite_scope_integration','Unreviewed package')
  require(review['producer_receipt_sha256']==item['receipt'] and rr['producer_receipt_sha256']==item['receipt'] and pr['producer_receipt_sha256']==item['receipt'],'Receipt linkage mismatch')
  require(rr['review_sha256']==item['review'] and review['peer_receipt_sha256']==item['peer_receipt_sha256'] and review['peer_review_sha256']==item['peer_review_sha256'],'Independent review linkage mismatch')
  require((pr.get('review')=={'path':'peer-review.json','sha256':item['peer_review_sha256']}) if item['task']!='cosine_compute' else pr.get('peer_review_sha256')==item['peer_review_sha256'],'Peer review reference mismatch')
  require(review['reviewed_candidate_source_sha256']==receipt['modules'] and len(receipt['modules'])==item['modules'],'Candidate module set mismatch')
  records=[]
  for origin,prefix,index in [(p,'producer',receipt['artifacts']), (v,'root-validation',rr['artifacts']), (w,'peer-review',pr['artifacts'])]:
   for rel,h in index.items():records.append((origin/relative(rel),'evidence/'+prefix+'/'+item['tag']+'/'+archive(rel),h))
  for path,h in pinned:
   origin,prefix=(p,'producer') if path.is_relative_to(p) else ((v,'root-validation') if path.is_relative_to(v) else (w,'peer-review'))
   records.append((path,'evidence/'+prefix+'/'+item['tag']+'/'+archive(path.relative_to(origin)),h))
  sources={}
  for rel,h in receipt['modules'].items():
   source=p/item['source_prefix']/relative(rel)
   require(receipt['checkpoint_module_source_paths'][rel]==rel+'.txt','Unexpected frozen module archive mapping')
   exact(source,h);sources[rel]=(source,h)
   records.append((source,'evidence/producer/'+item['tag']+'/'+rel+'.txt',h))
  for source,rel,h in records:
   if check_hashes:exact(source,h)
   require(source.stat().st_size<LIMIT,'Frozen file exceeds 100 MiB: '+str(source))
   require(rel not in destinations or destinations[rel]==h,'Conflicting archive destinations: '+rel)
   destinations[rel]=h
  delta=load(p/'evidence/numerical-authority-delta.json')
  require((len(delta['rules']),len(delta['provenance']),sum(len(t['cases']) for t in delta['requirements']))==(item['families'],item['ports'],item['cases']),'Delta counts mismatch')
  out.append({'spec':item,'producer':p,'root_review':v,'peer':w,'receipt':receipt,'review':review,'records':records,'sources':sources,'delta':delta})
 return out

def state(args):
 r,b,old=context(args);s=load(b/'evidence/preparation-state.json')
 active_gzip(r)
 require(s['deactivated_ledger_paths']==[] and s['historical_deletions_authorized'] is False,'Historical deletion or ledger migration is outside this batch')
 explicit_base(r,args.expected_base_commit)
 require(args.expected_base_commit==s['base_commit'],'Explicit base differs from preparation')
 require(args.scripts_receipt_sha256==s['scripts_receipt_sha256'],'Scripts receipt differs from preparation')
 require(args.storage_validation_sha256==s['storage_validation_sha256'],'Storage validation pin differs from preparation')
 verify_scripts(HERE,s['scripts_receipt_sha256']);verify_scripts(b/'evidence/integration-scripts',s['scripts_receipt_sha256'])
 require(digest(HERE/'specs.json')==s['specs_sha256'],'Integration specs changed')
 for flag,key in [('baseline_ledger','baseline_ledger_path'),('ledger','ledger_path'),('predecessor_catalog','predecessor_catalog'),('predecessor_supplement','predecessor_supplement'),('predecessor_review','predecessor_review')]:
  require(getattr(args,flag)==s[key],'Runtime path argument differs from preparation: '+flag)
 for rel,h in s['script_sha256'].items():exact(HERE/rel,h)
 require(hashes(r/STORAGE_CHECKPOINT)==s['storage_checkpoint_sha256'],'Committed gzip storage checkpoint changed')
 exact(b/s['baseline_ledger_archive'],s['baseline_ledger_sha256'])
 for rel,h in s['baseline_document_sha256'].items():exact(b/s['baseline_document_archives'][rel],h)
 for rel,h in s['protected_sha256'].items():exact(r/rel,h)
 require(hashes(old)==s['predecessor_sha256'],'Predecessor checkpoint changed')
 require(hashes(r/BASE)==s['metadata_baseline_sha256'],'Historical metadata baseline changed')
 for rel,h in s['new_source_sha256'].items():exact(r/rel,h)
 changed={x.decode() for command in [('diff','--name-only','-z','--no-renames','HEAD'),('ls-files','--others','--exclude-standard','-z')] for x in git(r,*command).split(b'\0') if x}
 allowed=active_paths(args,s);prefix=b.relative_to(r).as_posix()+'/'
 require(all(rel in allowed or rel.startswith(prefix) for rel in changed),'Unrelated worktree change outside exact integration paths')
 return r,b,old,s

def checkpoint_files(b):return {p for p in b.rglob('*') if p.is_file()}

def active_paths(args,s):
 return {args.ledger, *s['deactivated_ledger_paths'], QUEUE, 'bench/tritonbench_g/deferred_numerical_experiments.md',
  '.github/workflows/bench-audit.yml','bench/original_program_scope.md',
  'bench/reviewed_numerical_extensions.py','bench/tests/test_exact_numerical_bindings.py','bench/current_source_acceptance.md',
  *s['new_source_sha256'],*[p['adapter'] for p in SPEC['packages']]}

def preservation(args, include_ledger=True):
 r,b,old,s=state(args);ac=bench_imports(r)
 from current_source_catalog import validate_catalog,validate_supplement
 from current_value_metadata import check_supplement_scope
 from exact_numerical_bindings import collect,validate_exact_queue
 cat=ac.load(b/'current-source-catalog.json.gz');supp=ac.load(b/'catalog-supplement.json.gz')
 oc=ac.load(old/args.predecessor_catalog);osupp=ac.load(old/args.predecessor_supplement)
 validate_catalog(r,cat);validate_supplement(r,r/'bench/review-checkpoints',supp,set(cat['expected_task_names']));check_supplement_scope(supp,r/'bench/review-checkpoints')
 claims=lambda c:{(t['task'],d['file'],d['declaration']):d for t in c['tasks'] for d in t['declarations']}
 before,after=claims(oc),claims(cat)
 require(len(before)==BASELINE['public_declarations'] and len(after)==EXPECTED['public_declarations'] and all(after[k]==v for k,v in before.items()),'Existing public records changed')
 require(all(cat['registered_modules'].get(k)==v for k,v in oc['registered_modules'].items()),'Old module registration changed')
 require(set(cat['registered_modules'])-set(oc['registered_modules'])=={ac.module_name(p) for p in s['new_source_sha256']},'Unexpected module registration')
 require(len(cat['registered_modules'])==EXPECTED['registered_modules'] and cat['summary']['task_modules']==EXPECTED['task_modules'],'Catalog totals differ')
 require(sum(len(t['declarations']) for t in supp['tasks'])==EXPECTED['supplemental_contracts'],'Supplement total differs')
 require(Counter(d['role'] for t in supp['tasks'] for d in t['declarations'])=={'parametric_value_contract':EXPECTED['supplemental_execution_contracts'],'protocol_contract':EXPECTED['supplemental_protocol_contracts']},'Supplement role counts differ')
 sr={t['task']:t for t in supp['tasks']};osr={t['task']:t for t in osupp['tasks']}
 require(set(sr)-set(osr)==NEW_TASKS-set(osr),'Unexpected supplement tasks')
 for name,previous in osr.items():
  if name in NEW_TASKS:require(sr[name]['declarations'][:len(previous['declarations'])]==previous['declarations'],'Old supplemental declarations changed')
  else:require(sr[name]==previous,'Unrelated supplement row changed')
 semantic=ac.load(b/'root-review.json.gz')['semantic_review']
 for item in SPEC['packages']:
  proposal=ac.load(b/'evidence/producer'/item['tag']/'evidence/declaration-proposal.json');expected=deepcopy(proposal);expected['review_basis']=semantic[item['task']]
  declarations=[d for d in sr[item['task']]['declarations'] if (d['file'],d['declaration'])==(proposal['file'],proposal['declaration'])]
  require(declarations==[expected],'New public declaration/cases differ beyond root review attribution')
 legacy,authority,queue=collect(r);prior=ac.load(b/'evidence/baseline-queue.json.gz')
 require(prior==json.loads(gzip.decompress(git(r,'show',s['base_commit']+':'+QUEUE))), 'Queue snapshot differs from committed baseline')
 require(queue==ac.load(r/QUEUE),'Written queue differs from current authority')
 require(queue['rules'][:BASELINE['rules']]==prior['rules'] and queue['exact_callback_requirements'][:BASELINE['tables']]==prior['exact_callback_requirements'],'Old queue rule/table prefix changed')
 require(len(prior['exact_binding_provenance'])==BASELINE['bindings'] and all(queue['exact_binding_provenance'].get(k)==v for k,v in prior['exact_binding_provenance'].items()),'Old exact provenance changed')
 require(all(queue['exact_numerical_inputs'].get(k)==v for k,v in prior['exact_numerical_inputs'].items()),'Old numerical evidence inputs changed or deleted')
 new_rules=[];new_tables=[];new_provenance={}
 for item in SPEC['packages']:
  delta=ac.load(b/'evidence/producer'/item['tag']/'evidence/numerical-authority-delta.json')
  new_rules+=delta['rules'];new_tables+=delta['requirements'];new_provenance.update(delta['provenance'])
 require(queue['rules'][BASELINE['rules']:]==new_rules and queue['exact_callback_requirements'][BASELINE['tables']:]==new_tables,'New rules/cases differ from frozen reviewed deltas')
 require(queue['exact_binding_provenance']=={**prior['exact_binding_provenance'],**new_provenance},'Unexpected exact provenance addition or change')
 result=validate_exact_queue(queue,authority,legacy)
 require(result=={'status':'passed','exact_bindings':EXPECTED['exact_bindings'],'declaration_cases':EXPECTED['declaration_cases'],'native_executions':0,'structural_acceptance':False},'Exact authority counts differ')
 require(len(queue['rules'])==EXPECTED['numerical_rules'] and len(queue['exact_callback_requirements'])==EXPECTED['numerical_declarations'],'Queue totals differ')
 require((r/QUEUE).stat().st_size<LIMIT,'Compressed queue exceeds 100 MiB; stop before staging')
 if include_ledger:
  from check_original_program_scope import task_bindings,validate_task
  ledger=ac.load(r/args.ledger);prior=ac.load(b/'evidence/baseline-ledger.json.gz');pRows={t['task']:t for t in prior['tasks']};cRows={t['task']:t for t in cat['tasks']}
  baseline_raw=git(r,'show',s['base_commit']+':'+args.baseline_ledger)
  if args.baseline_ledger.endswith('.json.gz'):baseline_raw=gzip.decompress(baseline_raw)
  require(prior==json.loads(baseline_raw), 'Ledger snapshot differs from committed baseline')
  require(ledger['catalog_sha256']==ac.sha(ac.canonical(cat)) and ledger['deferred_queue']=={'path':QUEUE,'sha256':ac.hash_file(r/QUEUE)},'Ledger catalog/queue binding differs')
  unchanged=0
  for row in ledger['tasks']:
   name=row['task'];oldrow=pRows[name]
   require(row['input_sha256']==task_bindings(r,cat,cRows[name]),'Scope input bindings changed: '+name)
   if name in OLD_ACCEPTED:
    expected=deepcopy(oldrow);expected['review']=row['review'];expected['accepted_scope_progress']['scope_review']=row['review']
    require(row==expected,'Existing acceptance changed beyond catalog rebind: '+name)
   elif name not in NEW_TASKS:require(row==oldrow,'Unrelated scope row changed: '+name);unchanged+=1
   else:
    for key in ('accepted_scope_history','scope_reopening'):
     if key in oldrow:require(row[key]==oldrow[key],'Historical row state changed: '+name)
    item=next(p for p in SPEC['packages'] if p['task']==name);proposal=ac.load(b/'evidence/producer'/item['tag']/'evidence/scope-row-proposal.json')
    allowed={'status','input_sha256','proposed_status_after_independent_root_review','acceptance_boundary_review','review','accepted_scope_history','scope_reopening'}
    require(all(row.get(key)==value for key,value in proposal.items() if key not in allowed),'Frozen scope coverage/cases changed: '+name)
    require({k:v for k,v in row['acceptance_boundary_review'].items() if k not in {'status','queue_integration'}}=={k:v for k,v in proposal['acceptance_boundary_review'].items() if k not in {'status','queue_integration'}},'Frozen acceptance boundary changed: '+name)
   if name in NEW_TASKS|OLD_ACCEPTED:require(validate_task(r,cat,cRows[name],row,queue)==row['status'],'Scope validation differs: '+name)
  require(unchanged==172 and Counter(t['status'] for t in ledger['tasks'])=={'verified':1,'verified_with_deferred_numerics':10,'pending':172},'Scope progress differs')
 return {'catalog':cat,'supplement':supp,'queue':queue,'exact':result}

def verification_inputs(args):
 """Stable build/gate/test inputs; excludes generated logs and Lean objects."""
 r,b,old,s=state(args);ac=bench_imports(r);cat=ac.load(b/'current-source-catalog.json.gz')
 paths=set(PROTECTED)|active_paths(args,s)|{SOURCE_MAP,LEGACY,REVIEW}
 paths.update(cat['build_input_sha256']);paths.update(cat['python_input_sha256'])
 paths.update(ac.load(b/'root-review.json.gz')['reviewed_input_sha256'])
 paths.update(p.relative_to(r).as_posix() for folder in [r/'bench',r/'bench/tests',r/'scripts'] for p in folder.glob('*.py'))
 # Raw archive hashes cover every frozen receipt, source, attempt, object and log.
 paths.update((b/rel).relative_to(r).as_posix() for rel in ac.load(b/'evidence-index.json')['retained_exact_artifacts'])
 for name in ['current-source-catalog.json.gz','catalog-supplement.json.gz','root-review.json.gz','evidence-index.json','evidence/preparation-state.json','evidence/install-state.json','evidence/rebind-state.json']:
  if (b/name).exists():paths.add((b/name).relative_to(r).as_posix())
 for path in b.glob('*scope-review*.json.gz'):paths.add(path.relative_to(r).as_posix())
 return {rel:digest(r/relative(rel)) for rel in sorted(paths-set(s['deactivated_ledger_paths']))}
