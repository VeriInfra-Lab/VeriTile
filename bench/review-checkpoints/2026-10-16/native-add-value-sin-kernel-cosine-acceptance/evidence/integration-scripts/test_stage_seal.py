"""Bounded stage controls: /tmp filesystem and in-memory Git, never actual Git."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import sys,json,hashlib,contextlib,io
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent
O=Path(sys.argv[1]).resolve();O.mkdir(parents=True,exist_ok=False)
original=W.parent/'next-three-source-integration-scripts-wip/stage.py'
results=[]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def oid(p):
 b=Path(p).read_bytes();return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def fixture(name,mode,stage_source):
 root=O/name;cp=root/'bench/review-checkpoints/synthetic';cp.mkdir(parents=True)
 active={'bench/active-source.py','bench/tritonbench_g/deferred_numerical_experiments.json.gz'}
 for rel in active:
  p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic existing active input\n')
 known=cp/'evidence/frozen.txt';known.parent.mkdir();known.write_text('finalized evidence\n')
 state={'base_branch':'fp-examples','base_commit':'a'*40,'deactivated_ledger_paths':[]}
 scope={'requested':183,'accepted':11,'verified':1,'verified_with_deferred_numerics':10,'pending':172}
 validated={rel:digest(root/rel) for rel in active};validated[known.relative_to(root).as_posix()]=digest(known)
 validation={'scope':scope,'checked_input_sha256':validated,'deleted_paths':[],'current_comparator_success':False}
 (cp/'validation.json').write_text(json.dumps(validation))
 extra=cp/'post-finalize-unlisted.txt'
 if mode=='extra':extra.write_text('Added after final validation.\n')
 if mode=='missing':known.unlink()
 sim={'staged':set(),'added':False,'commands':[],'temps':[],'post_write_injected':False}
 def git(r,*args,input=None):
  assert r==root;sim['commands'].append(list(args))
  if args==('branch','--show-current'):return b'fp-examples\n'
  if args[:3]==('diff','--cached','--name-only'):return b'\0'.join(p.encode() for p in sorted(sim['staged']))+(b'\0' if sim['staged'] else b'')
  if args[:2]==('diff','--name-only'):return b'' if sim['added'] else b'\0'.join(p.encode() for p in sorted(active))+b'\0'
  if args[:1]==('status',):
   rows=[b' M '+p.encode() for p in sorted(active)]+[b'?? '+p.relative_to(root).as_posix().encode() for p in sorted(cp.rglob('*')) if p.is_file()]
   return b'\0'.join(rows)+b'\0'
  if args[:1]==('ls-tree',):return ('\n'.join(sorted(active))+'\n').encode()
  if args[:1]==('add',):
   named=next((a.split('=',1)[1] for a in args if a.startswith('--pathspec-from-file=')),None)
   if named:
    temp=Path(named);sim['temps'].append(temp);sim['staged'].update(p.decode() for p in temp.read_bytes().split(b'\0') if p)
   else:sim['staged'].add(args[-1])
   sim['added']=True;return b''
  if args[:1]==('ls-files',):return b''.join(('100644 '+('0'*40 if mode=='index_failure' else oid(root/p))+' 0\t'+p).encode()+b'\0' for p in sorted(sim['staged']))
  if args[:1]==('hash-object',):return ('\n'.join(oid(root/p) for p in input.decode().splitlines())+'\n').encode()
  if args[:1]==('show',):return (root/args[1][1:]).read_bytes()
  raise AssertionError(args)
 def diff_check(*args,**kwargs):
  if mode=='post_write_extra' and not sim['post_write_injected']:
   extra.write_text('Concurrent post-add checkpoint addition.\n');sim['post_write_injected']=True
  return SimpleNamespace(returncode=0,stdout='',stderr='')
 common=ModuleType('common');common.Path=Path;common.json=json;common.LIMIT=100*1024*1024;common.QUEUE='bench/tritonbench_g/deferred_numerical_experiments.json.gz'
 common.arguments=lambda _:SimpleNamespace(parse_args=lambda:SimpleNamespace())
 common.state=lambda a:(root,cp,None,state);common.load=lambda p:json.loads(Path(p).read_text());common.require=require;common.git=git
 common.exact=lambda p,h:require(Path(p).is_file() and digest(p)==h,'Changed/missing input: '+str(p))
 common.active_paths=lambda a,s:active;common.checkpoint_files=lambda b:{p for p in b.rglob('*') if p.is_file()};common.relative=lambda p:Path(p);common.archive=lambda p:str(p);common.digest=digest
 common.subprocess=SimpleNamespace(run=diff_check);common.write=lambda p,v:Path(p).write_text(json.dumps(v));sys.modules['common']=common
 def execute():
  stdout=io.StringIO();error=None
  try:
   with contextlib.redirect_stdout(stdout):exec(compile(stage_source.read_text(),str(stage_source),'exec'),{'__name__':'synthetic_stage_replay','__file__':str(stage_source)})
  except ValueError as e:error=str(e)
  return error,stdout.getvalue()
 return root,cp,extra,sim,execute
# Preserve the exact revision1 trigger, then require rejection under revision2.
root,cp,extra,sim,run=fixture('old-extra','extra',original);err,out=run();assert err is None and extra.relative_to(root).as_posix() in sim['staged'];results.append({'test':'revision1_unsealed_file_counterexample','result':'reproduced'})
for p in sim['temps']:p.unlink()
root,cp,extra,sim,run=fixture('new-baseline','baseline',W/'stage.py');err,out=run();assert err is None
assert all('--no-renames' in c for c in sim['commands'] if c[0]=='status' or (c[0]=='diff' and '--name-only' in c))
results.extend([{'test':'revision2_legal_baseline','result':'passed'},{'test':'stage_path_set_commands_disable_rename_detection','result':'passed'}])
for p in sim['temps']:p.unlink()
for mode,needle in [('extra','Checkpoint file set differs'),('missing','Changed/missing input')]:
 root,cp,extra,sim,run=fixture('new-'+mode,mode,W/'stage.py');err,out=run();assert needle in err and not sim['added'];results.append({'test':'revision2_'+mode+'_sealed_file_rejected_before_add','result':'passed','reason':err})
root,cp,extra,sim,run=fixture('new-index-failure','index_failure',W/'stage.py');err,out=run();assert err=='Index/worktree object mismatch' and sim['staged']
assert not (cp/'staging-review.json').exists();first=list(sim['commands']);retry,out=run();assert 'Index must be empty' in retry and sim['staged'];assert not any(c[0] in ('reset','restore','checkout','rm') for c in sim['commands'])
results.append({'test':'post_add_failure_retains_index_and_retry_requires_explicit_repair','result':'passed','first_error':err,'retry_error':retry})
for p in sim['temps']:p.unlink()
root,cp,extra,sim,run=fixture('new-post-write-extra','post_write_extra',W/'stage.py');err,out=run();assert err=='Checkpoint file set changed while staging' and (cp/'staging-review.json').exists() and sim['staged']
retry,out=run();assert retry=='Staging review already exists';results.append({'test':'post_add_file_set_change_rejects_and_existing_review_blocks_retry','result':'passed','first_error':err,'retry_error':retry})
for p in sim['temps']:p.unlink()
assert "('diff','--name-only','-z','--no-renames','HEAD')" in (W/'common.py').read_text();results.append({'test':'state_path_set_comparison_disables_rename_detection','result':'passed'})
report={'status':'passed','test_count':len(results),'tests':results,'source_sha256':{str(p):digest(p) for p in [original,W/'stage.py',W/'common.py']},'actual_git_commands_run':0,'actual_repository_modified':False,'producer_revision1_modified':False,'root_approval_issued':False,'acceptance_increment':0,'boundary':'Common.state is represented as an already-validated precondition; Git and git-diff subprocesses are in-memory simulations. Source/state/integration predicates are unchanged except literal path enumeration.'}
(O/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','test_count':len(results),'actual_git_commands_run':0}))
