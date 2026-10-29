"""Reviewed storage migration stages. No stage stages or commits Git changes."""
from pathlib import Path,PurePosixPath
from collections import Counter
from copy import deepcopy
import argparse,gzip,hashlib,json,os,re,shutil,subprocess,sys,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PLAIN='bench/tritonbench_g/deferred_numerical_experiments.json';COMPRESSED=PLAIN+'.gz';MARKDOWN=PLAIN[:-5]+'.md';LEDGER='bench/original_program_scope.json.gz'
POLICY='bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/primary-version-policy.md'
MAPPING='bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/three-version-inventory/eight-task-mapping.md'
def require(ok,message):
 if not ok:raise ValueError(message)
def digest(data):return hashlib.sha256(data).hexdigest()
def file_hash(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def text_bytes(value):return (json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode()
def unique(pairs):
 out={}
 for key,value in pairs:require(key not in out,'duplicate JSON key '+key);out[key]=value
 return out
def load(path):
 path=Path(path)
 with (gzip.open(path,'rt',encoding='utf-8') if path.name.endswith('.gz') else path.open(encoding='utf-8')) as f:return json.load(f,object_pairs_hook=unique)
def relative(root,name):
 p=PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts and str(p)==name and '\\' not in name,'unsafe relative path')
 result=Path(root)/name;require(result.resolve().is_relative_to(Path(root).resolve()),'escaping path '+name);return result
def atomic_bytes(path,raw):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.migration-tmp');require(not tmp.exists(),'unfinished write '+str(tmp));tmp.write_bytes(raw);os.replace(tmp,path)
def write(path,value):
 path=Path(path);raw=text_bytes(value)
 if path.name.endswith('.gz'):
  import io
  stream=io.BytesIO()
  with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as out:out.write(raw)
  raw=stream.getvalue()
 atomic_bytes(path,raw)
def compressed_bytes(path,raw):
 import io
 stream=io.BytesIO()
 with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as out:out.write(raw)
 atomic_bytes(path,stream.getvalue())
def archive_plain(source,target):
 source=Path(source);target=Path(target);require(not target.exists(),'archive already exists');target.parent.mkdir(parents=True,exist_ok=True)
 before=file_hash(source);size=source.stat().st_size
 with source.open('rb') as src,target.open('wb') as dst:
  with gzip.GzipFile(filename='',mode='wb',fileobj=dst,mtime=0) as stream:shutil.copyfileobj(src,stream,1024*1024)
 require(decompressed_hash(target)==before and file_hash(source)==before,'archive changed bytes')
 return {'path':str(target),'compressed_sha256':file_hash(target),'decompressed_sha256':before,'decompressed_bytes':size}
def decompressed_hash(path):
 h=hashlib.sha256()
 with gzip.open(path,'rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo).decode().strip()
def commit_check(repo,expected,clean=False):
 require(bool(re.fullmatch(r'[0-9a-f]{40}',expected)),'an explicit full expected base commit is required; no HEAD alias or hardcoded historical commit')
 require(git(repo,'rev-parse','HEAD')==expected,'HEAD differs from explicit expected base commit')
 require(git(repo,'cat-file','-t',expected)=='commit','base is not a commit')
 if clean:require(not git(repo,'status','--porcelain=v1','--untracked-files=all'),'base tree is dirty; finish and commit the current batch first')
def changed_paths(repo):
 tracked=git(repo,'diff','--name-only','HEAD').splitlines();other=git(repo,'ls-files','--others','--exclude-standard').splitlines();return set(tracked+other)
def command(run,name,argv,cwd):
 run=Path(run);log=run/(name+'.log');require(not log.exists(),'attempt already recorded: '+name);start=time.monotonic()
 with log.open('w') as f:r=subprocess.run(argv,cwd=cwd,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT)
 row={'command':list(map(str,argv)),'exit_code':r.returncode,'seconds':round(time.monotonic()-start,3),'log':log.name,'log_sha256':file_hash(log)};write(run/(name+'.command.json'),row);return row
def checkpoint(plan):return relative(Path(plan['repo']),plan['checkpoint'])
def external_run(repo,path):
 run=Path(path).resolve();require(not run.is_relative_to(Path(repo).resolve()),'run output must be outside the repository and its checkpoint');return run
def verify_script_package(root,expected):
 root=Path(root);p=root/'receipt.json';require(p.is_file() and file_hash(p)==expected,'integration script receipt mismatch');receipt=load(p)
 require(isinstance(receipt.get('artifacts'),dict) and bool(receipt['artifacts']) and 'receipt.json' not in receipt['artifacts'],'invalid integration script artifact closure')
 for rel,h in receipt['artifacts'].items():require(file_hash(relative(root,rel))==h,'integration candidate changed: '+rel)
 return receipt
def verify_candidate(expected):return verify_script_package(HERE,expected)
def copy_script_package(target,expected):
 receipt=verify_candidate(expected);target=Path(target);require(not target.exists(),'integration script archive already exists');target.mkdir(parents=True)
 pins={'receipt.json':expected,**receipt['artifacts']}
 for rel,h in pins.items():
  source=relative(HERE,rel);dest=relative(target,rel);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
  require(file_hash(dest)==h and file_hash(source)==h,'integration script copy changed: '+rel)
 verify_script_package(target,expected)
 return {'path':'evidence/integration-scripts','receipt_sha256':expected,'artifact_count':len(receipt['artifacts']),'copied_files':len(pins),'copied_bytes':sum(relative(target,rel).stat().st_size for rel in pins)}
def read_plan(args):
 require(file_hash(args.plan)==args.plan_sha256,'plan hash mismatch');plan=load(args.plan);repo=Path(plan['repo']);external_run(repo,plan['run_directory']);verify_candidate(plan['scripts_receipt_sha256']);commit_check(repo,plan['expected_base_commit'])
 require(args.expected_base_commit==plan['expected_base_commit'],'stage base commit differs from reviewed plan')
 allowed=set(plan['mutation_paths']);unexpected={p for p in changed_paths(repo) if p not in allowed and not p.startswith(plan['checkpoint']+'/')}
 require(not unexpected,'unrelated worktree mutation after planning: '+str(sorted(unexpected)))
 for rel,h in plan['unchanged_inputs'].items():require(file_hash(repo/rel)==h,'unchanged input drift: '+rel)
 return plan
def stage_record(plan,name,data):
 run=Path(plan['run_directory']);p=run/(name+'.json');require(not p.exists(),'completed/failed stage already exists; inspect it before retrying');write(p,data);return p
def stage_input(plan,name):
 p=Path(plan['run_directory'])/(name+'.json');r=load(p);require(r['status']=='passed','previous stage did not pass: '+name);return r
def render_doc(rel,data,cp):
 return data.replace(b'@MIGRATION_CHECKPOINT@',cp.removeprefix('bench/').encode()) if rel.endswith('.md') else data
def plan_stage(args):
 repo=args.repo.resolve();run=external_run(repo,args.output);commit_check(repo,args.expected_base_commit,clean=True);verify_candidate(args.scripts_receipt_sha256)
 cp=args.checkpoint;relative(repo,cp);require(cp.startswith('bench/review-checkpoints/') and not (repo/cp).exists(),'checkpoint must be a fresh review-checkpoint directory')
 source=load(HERE/'evidence/source-plan.json');changes=source['changes'];plain=repo/PLAIN;require(plain.is_file() and not (repo/COMPRESSED).exists() and not (repo/COMPRESSED).is_symlink(),'expected exactly one plain pre-migration queue')
 for rel,row in changes.items():
  target=repo/rel;require((file_hash(target) if target.exists() else None)==row['before'],'candidate requires delta review for changed preimage: '+rel)
 for rel,h in source['unchanged_consumers'].items():require(file_hash(repo/rel)==h,'consumer drift requires delta review: '+rel)
 for rel,h in source['policy_inputs'].items():require(file_hash(repo/rel)==h,'policy/map changed: '+rel)
 catalog=relative(repo,args.catalog);ledger=load(repo/LEDGER);require(ledger['deferred_queue']=={'path':PLAIN,'sha256':file_hash(plain)},'legacy queue reference is not exact')
 require(ledger['catalog_sha256']==digest(canonical(load(catalog))),'catalog/ledger mismatch')
 require(len(ledger['tasks'])==183,'wrong legacy task population')
 require(not run.exists(),'plan output must be fresh');run.mkdir(parents=True)
 # Real before/after comparisons must be captured against this explicit committed base.
 q=command(run,'baseline-queue-check',[sys.executable,'-B',str(repo/'bench/deferred_numerical_experiments.py'),'--check'],repo);require(q['exit_code']==0,'base queue does not reproduce')
 scope=command(run,'baseline-scope-check',[sys.executable,'-B',str(repo/'bench/check_original_program_scope.py'),'--root',str(repo),'--catalog',str(catalog),'--ledger',str(repo/LEDGER),'--output',str(run/'baseline-scope.json')],repo)
 require(scope['exit_code'] in [0,1] and (run/'baseline-scope.json').exists(),'base scope checker failed to produce report');report=load(run/'baseline-scope.json');require(report.get('requested_tasks')==183 and len(report.get('tasks',[]))==183,'invalid base scope report')
 expected_status={t['task']:t['status'] for t in ledger['tasks']}
 for t in report['tasks']:
  if expected_status[t['task']] in ['verified','verified_with_deferred_numerics']:require(t['status']==expected_status[t['task']],'previously accepted legacy task failed its base check')
 commit_check(repo,args.expected_base_commit,clean=True)
 replacements={rel:{'before':row['before'],'after':digest(render_doc(rel,(HERE/'after'/rel).read_bytes(),cp)),'candidate_source':'after/'+rel} for rel,row in changes.items()}
 unchanged={**source['unchanged_consumers'],**source['policy_inputs'],args.catalog:file_hash(catalog)}
 plan={'schema':1,'kind':'queue_gzip_reviewable_installation_plan','repo':str(repo),'expected_base_commit':args.expected_base_commit,'base_commit_is_pre_migration':True,'run_directory':str(run),'checkpoint':cp,'scripts_receipt_sha256':args.scripts_receipt_sha256,'catalog':args.catalog,'ledger':LEDGER,'old_queue':{'path':PLAIN,'sha256':file_hash(plain),'bytes':plain.stat().st_size},'ledger_before_sha256':file_hash(repo/LEDGER),'legacy_task_rows_sha256':digest(canonical(ledger['tasks'])),'legacy_status_counts':dict(Counter(t['status'] for t in ledger['tasks'])),'baseline_scope_sha256':file_hash(run/'baseline-scope.json'),'replacements':replacements,'unchanged_inputs':unchanged,'mutation_paths':sorted([*replacements,PLAIN,COMPRESSED,MARKDOWN,LEDGER]),'ledger_fields_allowed_to_change':['/deferred_queue/path','/deferred_queue/sha256'],'primary_gate_installed':False,'acceptance_increment':0}
 write(run/'plan.json',plan);print(json.dumps({'plan':str(run/'plan.json'),'sha256':file_hash(run/'plan.json'),'expected_base_commit':args.expected_base_commit,'repository_modified':False}))
def install_stage(args):
 p=read_plan(args);repo=Path(p['repo']);commit_check(repo,p['expected_base_commit'],clean=True);cp=checkpoint(p);require(not cp.exists(),'checkpoint already exists')
 require(file_hash(repo/PLAIN)==p['old_queue']['sha256'] and file_hash(repo/LEDGER)==p['ledger_before_sha256'],'base documents changed')
 # Archive verification precedes any unlink or installed source replacement.
 archived=archive_plain(repo/PLAIN,cp/'before/active-queue.plain-json.gz')
 for rel,row in p['replacements'].items():
  target=repo/rel;require((file_hash(target) if target.exists() else None)==row['before'],'preimage changed '+rel)
  if target.exists():dest=cp/'before'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(target,dest)
 before=cp/'before/scope-ledger.json.gz';shutil.copyfile(repo/LEDGER,before);require(file_hash(before)==p['ledger_before_sha256'],'ledger archive changed')
 write(cp/'plan.json',p);scripts=copy_script_package(cp/'evidence/integration-scripts',p['scripts_receipt_sha256'])
 for rel,row in p['replacements'].items():
  data=render_doc(rel,(HERE/row['candidate_source']).read_bytes(),p['checkpoint']);require(digest(data)==row['after'],'candidate render changed');atomic_bytes(repo/rel,data)
 require(file_hash(repo/PLAIN)==archived['decompressed_sha256'] and decompressed_hash(cp/'before/active-queue.plain-json.gz')==p['old_queue']['sha256'],'archive verification failed before deactivation')
 (repo/PLAIN).unlink();stage_record(p,'install',{'status':'passed','archive':archived,'source_changes':p['replacements'],'integration_scripts':scripts,'active_plain_removed_after_archive_verification':True,'acceptance_increment':0});print('Installed source candidates and archived/deactivated plain queue. Generation is the next required stage.')
def verify_installed(p):
 repo=Path(p['repo']);stage_input(p,'install');verify_script_package(checkpoint(p)/'evidence/integration-scripts',p['scripts_receipt_sha256']);require(not (repo/PLAIN).exists() and not (repo/PLAIN).is_symlink(),'competing plain queue')
 for rel,row in p['replacements'].items():require(file_hash(repo/rel)==row['after'],'installed source changed '+rel)
def generate_stage(args):
 p=read_plan(args);verify_installed(p);repo=Path(p['repo']);run=Path(p['run_directory']);r=command(run,'generate-current-authority',[sys.executable,'-B',str(repo/'bench/deferred_numerical_experiments.py'),'--root',str(repo)],repo);require(r['exit_code']==0,'queue regeneration failed; preserved log and original compressed archive')
 require(decompressed_hash(repo/COMPRESSED)==p['old_queue']['sha256'],'authority or serialized task rows changed; storage-only migration must stop for review')
 stage_record(p,'generate',{'status':'passed','queue_sha256':file_hash(repo/COMPRESSED),'queue_bytes':(repo/COMPRESSED).stat().st_size,'decompressed_queue_sha256':p['old_queue']['sha256'],'all_queue_bytes_rules_tasks_bindings_callbacks_unchanged':True,'markdown_sha256':file_hash(repo/MARKDOWN),'acceptance_increment':0});print('Regenerated compressed queue is byte-for-byte identical after decompression.')
def rebind_ledger(old,queue_hash):
 require(old.get('deferred_queue',{}).get('path')==PLAIN,'unexpected prior ledger queue path');new=deepcopy(old);new['deferred_queue']={'path':COMPRESSED,'sha256':queue_hash}
 require(new['tasks']==old['tasks'],'task rows changed');require({k:v for k,v in new.items() if k!='deferred_queue'}=={k:v for k,v in old.items() if k!='deferred_queue'},'non-reference ledger fields changed');return new
def rebind_ledger_bytes(raw,queue_hash):
 old=json.loads(raw,object_pairs_hook=unique);new=rebind_ledger(old,queue_hash)
 def block(ref):return ('"deferred_queue": '+json.dumps(ref,indent=2,ensure_ascii=False).replace('\n','\n  ')).encode()
 before=block(old['deferred_queue']);after=block(new['deferred_queue']);require(raw.count(before)==1,'ledger queue block has unexpected formatting; review rather than reserialize task rows')
 result=raw.replace(before,after,1);require(json.loads(result,object_pairs_hook=unique)==new,'ledger byte replacement changed unrelated data');return result
def rebind_stage(args):
 p=read_plan(args);verify_installed(p);g=stage_input(p,'generate');repo=Path(p['repo']);require(file_hash(repo/COMPRESSED)==g['queue_sha256'] and file_hash(repo/LEDGER)==p['ledger_before_sha256'],'generated queue or original ledger changed')
 old=load(repo/LEDGER);require(digest(canonical(old['tasks']))==p['legacy_task_rows_sha256'],'prior task rows changed');new=rebind_ledger(old,g['queue_sha256']);raw=gzip.decompress((repo/LEDGER).read_bytes());rebound=rebind_ledger_bytes(raw,g['queue_sha256']);compressed_bytes(repo/LEDGER,rebound)
 stage_record(p,'rebind',{'status':'passed','ledger_sha256':file_hash(repo/LEDGER),'changed_fields':p['ledger_fields_allowed_to_change'],'legacy_task_rows_sha256':digest(canonical(new['tasks'])),'all_bytes_outside_deferred_queue_block_unchanged':True,'all_other_ledger_fields_unchanged':True,'acceptance_increment':0});print('Rebound only the queue path and exact compressed-file hash; all legacy rows are identical.')
def snapshot_stage(args):
 p=read_plan(args);verify_installed(p);g=stage_input(p,'generate');b=stage_input(p,'rebind');repo=Path(p['repo']);cp=checkpoint(p);out=cp/'snapshot';require(not out.exists(),'snapshot already exists')
 require(file_hash(repo/COMPRESSED)==g['queue_sha256'] and file_hash(repo/LEDGER)==b['ledger_sha256'],'snapshot documents drifted');cat=load(repo/p['catalog']);ledger=load(repo/LEDGER);queue=load(repo/COMPRESSED)
 require(ledger['catalog_sha256']==digest(canonical(cat)),'mixed catalog/ledger snapshot');require(ledger['deferred_queue']=={'path':COMPRESSED,'sha256':file_hash(repo/COMPRESSED)},'mixed ledger/queue snapshot');require(digest(canonical(ledger['tasks']))==p['legacy_task_rows_sha256'],'task row drift')
 sys.path.insert(0,str(repo/'bench'));from exact_numerical_bindings import PACKAGE_PATHS
 pins={}
 def add(rel,h):
  require(rel not in pins or pins[rel]==h,'conflicting snapshot pin '+rel);require(file_hash(relative(repo,rel))==h,'snapshot input drift '+rel);pins[rel]=h
 for field in ['build_input_sha256','python_input_sha256','source_variant_evidence_sha256']:
  for rel,h in cat[field].items():add(rel,h)
 for rel,row in cat['python_sources'].items():
  if row.get('sha256') and rel not in cat['missing_bound_python_sources']:add(rel,row['sha256'])
 for rel,h in queue['input_sha256'].items():add(rel,h)
 for key,h in queue['exact_numerical_inputs'].items():
  package,rel=key.split(':',1);require(package in PACKAGE_PATHS,'unknown exact-authority package');add((PACKAGE_PATHS[package]+'/' if PACKAGE_PATHS[package] else '')+rel,h)
 for row in ledger['tasks']:
  if row.get('review'):add(row['review']['path'],row['review']['sha256'])
 for row in cat['tasks']:
  if row.get('metadata_correction_review'):r=row['metadata_correction_review'];add(r['path'],r['sha256'])
 helpers=sorted({*repo.glob('bench/*.py'),*(repo/'bench/current_catalog_vendor').rglob('*.py'),*(repo/'scripts').rglob('*.py')})
 for file in helpers:add(file.relative_to(repo).as_posix(),file_hash(file))
 for rel in [p['catalog'],LEDGER,COMPRESSED,POLICY,MAPPING,'.github/workflows/bench-audit.yml']:add(rel,file_hash(repo/rel))
 # The base is a real explicit commit. Overlay documents are labelled as worktree output.
 tree={}
 for line in git(repo,'ls-tree','-r',p['expected_base_commit']).splitlines():
  header,rel=line.split('\t',1);mode,kind,oid=header.split()
  if kind=='blob':tree[rel]=oid
 overlays={rel for rel in pins if rel in p['mutation_paths']};require(set(pins)-overlays<=set(tree),'non-overlay snapshot source is absent from committed base')
 origins={rel:{'kind':'reviewed_storage_overlay','base_blob_oid':tree.get(rel)} if rel in overlays else {'kind':'committed_base_blob','blob_oid':tree[rel]} for rel in pins}
 out.mkdir(parents=True);documents={}
 for label,rel in [('catalog',p['catalog']),('legacy',LEDGER),('queue',COMPRESSED)]:
  target=out/(label+'.json.gz')
  if rel.endswith('.gz'):shutil.copyfile(repo/rel,target)
  else:archive_plain(repo/rel,target)
  documents[label]={'path':target.name,'sha256':file_hash(target),'source_path':rel,'source_file_sha256':file_hash(repo/rel),'decompressed_sha256':decompressed_hash(target),'origin':origins[rel]}
 manifest={'schema':1,'kind':'coherent_post_migration_snapshot_for_future_primary_sidecar','base_commit':p['expected_base_commit'],'base_commit_is_pre_migration':True,'post_migration_commit':None,'migration_plan_sha256':file_hash(Path(p['run_directory'])/'plan.json'),'documents':documents,'root_input_sha256':pins,'root_input_origins':origins,'helper_sha256':{f.relative_to(repo).as_posix():file_hash(f) for f in helpers},'policy':{'path':POLICY,'sha256':pins[POLICY]},'eight_task_mapping':{'path':MAPPING,'sha256':pins[MAPPING]},'legacy_task_rows_sha256':p['legacy_task_rows_sha256'],'legacy_status_counts':p['legacy_status_counts'],'primary_checker_installed':False,'primary_version_statuses_inferred':False,'future_sidecar_note':'Refresh evidence inventory and helper pins from this snapshot in a separately reviewed primary-gate delta; do not reuse the prototype fixed 97e11a9 snapshot or rewrite its frozen receipts.','acceptance_increment':0}
 write(out/'manifest.json.gz',manifest);stage_record(p,'snapshot',{'status':'passed','manifest':str((out/'manifest.json.gz').relative_to(repo)),'sha256':file_hash(out/'manifest.json.gz'),'root_input_count':len(pins),'dependency_files_copied':0,'acceptance_increment':0});print('Captured coherent base-plus-storage-overlay snapshot; no primary gate or new completion counts installed.')
def check_stage(args):
 p=read_plan(args);verify_installed(p);g=stage_input(p,'generate');b=stage_input(p,'rebind');s=stage_input(p,'snapshot');repo=Path(p['repo']);run=Path(p['run_directory']);require(file_hash(repo/COMPRESSED)==g['queue_sha256'] and file_hash(repo/LEDGER)==b['ledger_sha256'],'active documents changed')
 require(decompressed_hash(repo/COMPRESSED)==p['old_queue']['sha256'],'queue bytes changed');old=load(checkpoint(p)/'before/scope-ledger.json.gz');require(load(repo/LEDGER)==rebind_ledger(old,g['queue_sha256']),'ledger delta exceeds reference fields')
 tests=command(run,'storage-and-scope-regressions',[sys.executable,'-B','-m','unittest','discover','-s','bench/tests','-p','test_numerical_queue*.py'],repo);require(tests['exit_code']==0,'storage/scope regressions failed')
 q=command(run,'post-queue-check',[sys.executable,'-B',str(repo/'bench/deferred_numerical_experiments.py'),'--root',str(repo),'--check'],repo);require(q['exit_code']==0,'active queue --check failed')
 scope=command(run,'post-scope-check',[sys.executable,'-B',str(repo/'bench/check_original_program_scope.py'),'--root',str(repo),'--catalog',str(repo/p['catalog']),'--ledger',str(repo/LEDGER),'--output',str(run/'post-scope.json')],repo);require(scope['exit_code'] in [0,1] and load(run/'post-scope.json')==load(run/'baseline-scope.json'),'legacy scope outcomes changed')
 manifest=load(repo/s['manifest']);require(file_hash(repo/s['manifest'])==s['sha256'],'snapshot manifest changed')
 for rel,h in manifest['root_input_sha256'].items():require(file_hash(repo/rel)==h,'snapshot pin changed: '+rel)
 for ref in manifest['documents'].values():require(file_hash((repo/s['manifest']).parent/ref['path'])==ref['sha256'],'snapshot document changed')
 for rel in changed_paths(repo):
  file=repo/rel
  if file.is_file():require(file.stat().st_size<100*1024*1024,'new/changed Git file >=100 MiB: '+rel)
 data={'status':'passed','migration_storage_and_reference_checks':'passed','queue_bytes_and_all_legacy_rows_preserved':True,'legacy_scope_result_unchanged':True,'legacy_scope_status':load(run/'post-scope.json')['status'],'CI_deferred_numerics_gate_preserved':True,'coherent_snapshot':s,'full_global_build_comparator_gates':'not_run_by_storage_script; report separately if actually run','primary_gate_installed':False,'acceptance_increment':0};stage_record(p,'check',data)
 # Preserve exact commands/results; the checkpoint and run directories remain reviewable.
 for file in run.iterdir():
  if file.is_file():dest=checkpoint(p)/'validation'/file.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
 print(json.dumps(data))
def main():
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
 a=sub.add_parser('plan');a.add_argument('--repo',type=Path,required=True);a.add_argument('--expected-base-commit',required=True);a.add_argument('--scripts-receipt-sha256',required=True);a.add_argument('--catalog',required=True);a.add_argument('--checkpoint',required=True);a.add_argument('--output',type=Path,required=True)
 for name in ['install','generate','rebind','snapshot','check']:
  a=sub.add_parser(name);a.add_argument('--plan',type=Path,required=True);a.add_argument('--plan-sha256',required=True);a.add_argument('--expected-base-commit',required=True)
 args=parser.parse_args();globals()[args.stage+'_stage'](args)
if __name__=='__main__':main()
