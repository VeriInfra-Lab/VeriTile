"""Candidate integration helpers. Importing this module never mutates the repo."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import argparse, ast, gzip, hashlib, importlib.util, json, os, shutil, subprocess, sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / 'specs.json').read_text())
EXPECTED = SPEC['expected_after']
LIMIT = 100 * 1024 * 1024
GZIP_RECEIPT_SHA256 = '48b2f06eb4b0194e8a13c7c03fda82e9ca50cf88f1a4fd9701fc854ad9b6ef0e'
STORAGE_CHECKPOINT = 'bench/review-checkpoints/2026-10-16/explicit-json-gzip-storage'
QUEUE = 'bench/tritonbench_g/deferred_numerical_experiments.json'
PROTECTED = [
 'bench/adapter_common.py', 'bench/check_original_program_scope.py',
 'bench/current_source_catalog.py', 'bench/current_source_audit.py',
 'bench/current_value_metadata.py', 'bench/exact_numerical_bindings.py',
 'bench/exact_scope_gate.py', 'bench/deferred_numerical_experiments.py',
 'lakefile.toml', 'lake-manifest.json', 'lean-toolchain',
 'bench/original_program_acceptance.md',
]
BASE = 'bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline'
SOURCE_MAP = 'bench/current_source_selections.json'
LEGACY = 'bench/tritonbench_g/current_completion_inventory.json'
REVIEW = 'bench/review-checkpoints/2026-10-15/raw-copy-software-metadata/catalog-review-rebased.json'
TARGETS = ['VeriTile','VeriTileMath','VeriTileFull','TritonBenchGVerified',
 'TritonBenchGInProgress','TritonBenchGLegacy','TritonBenchSpecExamples','FPFrontendRegressions']
OLD_ACCEPTED = {'matrix_transpose','quantize_global','vector_addition','quant_transpose_kernel'}
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
 p=Path(path);require(not p.is_absolute() and '..' not in p.parts,'Unsafe relative path: '+str(p))
 return p

def archive(path):
 text=relative(path).as_posix()
 if text.endswith('.lean'):return text+'.txt'
 if text.endswith(('.olean','.olean.private','.olean.server','.ilean','.ir','.pyc')):return text+'.bin'
 return text

def hashes(root):return {p.relative_to(root).as_posix():digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}
def prose(value):return value if isinstance(value,str) else '\n'.join(k+': '+(v if isinstance(v,str) else json.dumps(v,sort_keys=True)) for k,v in value.items())
def git(root,*args,input=None):return subprocess.check_output(['git',*args],cwd=root,input=input)

def arguments(description):
 p=argparse.ArgumentParser(description=description)
 p.add_argument('--root',type=Path,default=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile'))
 p.add_argument('--packages',type=Path,default=HERE.parent)
 # These are explicit names, never extension or existence fallbacks.
 p.add_argument('--ledger',default='bench/original_program_scope.json.gz')
 p.add_argument('--baseline-ledger',default='bench/original_program_scope.json')
 p.add_argument('--predecessor-catalog',default='current-source-catalog.json')
 p.add_argument('--predecessor-supplement',default='catalog-supplement.json')
 p.add_argument('--predecessor-review',default='root-review.json')
 return p

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
  require(pr['review']=={'path':'peer-review.json','sha256':item['peer_review_sha256']},'Peer review reference mismatch')
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

def estimated_queue(root,packages):
 """Conservative layout estimate, not an authority validation receipt."""
 q=deepcopy(load(root/QUEUE));tables=q['exact_callback_requirements']
 for p in packages:
  item=p['spec'];d=p['delta'];old={r['id'] for r in q['rules']}
  require(not old.intersection(r['id'] for r in d['rules']),'Delta rule collision')
  q['rules']+=deepcopy(d['rules']);q['exact_binding_provenance'].update(deepcopy(d['provenance']));tables+=deepcopy(d['requirements'])
  for row in q['tasks']:
   if row['task']==item['task']:
    row['rule_ids'] += [r['id'] for r in d['rules']]
    row['exact_numerical_declarations']=[t['declaration'] for t in tables if t['task']==row['task']]
  # Deliberately include every producer/root/peer archive input, a superset of adapter bindings.
  for _,rel,h in p['records']:
   q['exact_numerical_inputs']['repo:bench/review-checkpoints/'+SPEC['intended_checkpoint']+'/'+rel]=h
  for rel,(_,h) in p['sources'].items():q['exact_numerical_inputs']['repo:'+rel]=h
  q['exact_numerical_summary'][item['tag']+'_estimated_delta']=d['summary']
 q['summary']['rule_profiles']=len(q['rules'])
 q['summary']['rule_statuses']=dict(sorted(Counter(r['queue_status'] for r in q['rules']).items()))
 q['exact_numerical_summary'].update(new_scalar_families=len(q['rules'])-11,scope_bindings=len(q['exact_binding_provenance']),declaration_cases=sum(len(t['cases']) for t in tables),registered_declarations=len(tables),unvalidated_numeric_instances=len(q['exact_binding_provenance']))
 # 1 MiB reserve covers new dependency keys and summary spelling differences.
 size=len((json.dumps(q,indent=2,ensure_ascii=False)+'\n').encode())
 return {'method':'current plain queue plus all frozen deltas and all archived input references; includes a 1 MiB additional reserve','estimated_serialized_bytes':size,'reserved_bytes':1048576,'conservative_bytes':size+1048576,'limit_bytes':LIMIT,'under_limit_with_reserve':size+1048576<LIMIT,'not_source_authority_validation':True}

def state(args):
 r,b,old=context(args);s=load(b/'evidence/preparation-state.json')
 require(git(r,'rev-parse','HEAD').decode().strip()==s['base_commit'],'HEAD changed since preparation')
 require(digest(HERE/'specs.json')==s['specs_sha256'],'Integration specs changed')
 for flag,key in [('baseline_ledger','baseline_ledger_path'),('ledger','ledger_path'),('predecessor_catalog','predecessor_catalog'),('predecessor_supplement','predecessor_supplement'),('predecessor_review','predecessor_review')]:
  require(getattr(args,flag)==s[key],'Runtime path argument differs from preparation: '+flag)
 for rel,h in s['script_sha256'].items():exact(HERE/rel,h)
 exact(HERE/'gzip-storage-receipt.json',GZIP_RECEIPT_SHA256)
 require(hashes(r/STORAGE_CHECKPOINT)==s['storage_checkpoint_sha256'],'Committed gzip storage checkpoint changed')
 exact(b/s['baseline_ledger_archive'],s['baseline_ledger_sha256'])
 for rel,h in s['baseline_document_sha256'].items():exact(b/s['baseline_document_archives'][rel],h)
 for rel,h in s['protected_sha256'].items():exact(r/rel,h)
 require(hashes(old)==s['predecessor_sha256'],'Predecessor checkpoint changed')
 require(hashes(r/BASE)==s['metadata_baseline_sha256'],'Historical metadata baseline changed')
 for rel,h in s['new_source_sha256'].items():exact(r/rel,h)
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
 require(len(before)==957 and len(after)==EXPECTED['public_declarations'] and all(after[k]==v for k,v in before.items()),'Existing public records changed')
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
 legacy,authority,queue=collect(r);prior=ac.load(b/'evidence/baseline-queue.json.gz')
 require(prior==json.loads(git(r,'show',s['base_commit']+':'+QUEUE)), 'Queue snapshot differs from committed baseline')
 require(queue==ac.load(r/QUEUE),'Written queue differs from current authority')
 require(queue['rules'][:153]==prior['rules'] and queue['exact_callback_requirements'][:8]==prior['exact_callback_requirements'],'Old queue rule/table prefix changed')
 require(len(prior['exact_binding_provenance'])==1642 and all(queue['exact_binding_provenance'].get(k)==v for k,v in prior['exact_binding_provenance'].items()),'Old exact provenance changed')
 result=validate_exact_queue(queue,authority,legacy)
 require(result=={'status':'passed','exact_bindings':EXPECTED['exact_bindings'],'declaration_cases':EXPECTED['declaration_cases'],'native_executions':0,'structural_acceptance':False},'Exact authority counts differ')
 require(len(queue['rules'])==EXPECTED['numerical_rules'] and len(queue['exact_callback_requirements'])==EXPECTED['numerical_declarations'],'Queue totals differ')
 require((r/QUEUE).stat().st_size<LIMIT,'Plain queue exceeds 100 MiB; stop before staging')
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
   if name in NEW_TASKS|OLD_ACCEPTED:require(validate_task(r,cat,cRows[name],row,queue)==row['status'],'Scope validation differs: '+name)
  require(unchanged==175 and Counter(t['status'] for t in ledger['tasks'])=={'verified':1,'verified_with_deferred_numerics':7,'pending':175},'Scope progress differs')
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
