from pathlib import Path
import ast,gzip,json,hashlib,difflib,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');P=W.parent/'queue-gzip-migration-wip'
sys.path.insert(0,str(W));import integration as m
source=m.load(W/'evidence/source-plan.json');checks={}
for rel,row in source['changes'].items():
 path=R/rel;m.require((m.file_hash(path) if path.exists() else None)==row['before'],'live preimage drift '+rel);m.require(m.file_hash(W/'after'/rel)==row['after'],'candidate source drift '+rel)
for rel,h in {**source['unchanged_consumers'],**source['policy_inputs']}.items():m.require(m.file_hash(R/rel)==h,'consumer/policy drift '+rel)
def defs(path):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
old=defs(P/'after/bench/deferred_numerical_experiments.py');new=defs(W/'after/bench/deferred_numerical_experiments.py');changed=[k for k in old if old[k]!=new.get(k)];m.require(changed==['write_outputs'],'unexpected producer AST change');checks['producer_authority_and_collection_AST_unchanged']=True
old=defs(P/'after/bench/check_original_program_scope.py');new=defs(W/'after/bench/check_original_program_scope.py');changed=[k for k in old if old[k]!=new.get(k)];m.require(changed==['check'] and set(new)-set(old)=={'load_active_queue'},'unexpected scope AST change');checks['scope_acceptance_predicate_ASTs_unchanged']=True
before=(P/'after/bench/check_original_program_scope.py').read_text();after=(W/'after/bench/check_original_program_scope.py').read_text();helper=after[after.index('\ndef load_active_queue('):after.index('\ndef main():')];after=after.replace(helper,'\n',1)
oldblock="    queue_ref = ledger.get('deferred_queue') or {}\n    if queue_ref.get('path') != QUEUE:\n        raise ValueError('scope ledger must bind the unified numerical queue')\n    queue = load(checked(root, QUEUE, queue_ref.get('sha256')))"
after=after.replace("    queue = load_active_queue(root, ledger.get('deferred_queue') or {})",oldblock)
m.require(ast.dump(ast.parse(after),include_attributes=False)==ast.dump(ast.parse(before),include_attributes=False),'scope changes exceed queue loader factoring')
workflow=(R/'.github/workflows/bench-audit.yml').read_text();runner=(R/'bench/current_source_audit.py').read_text();m.require('bench/current_source_audit.py' in workflow and '--scope-ledger bench/original_program_scope.json.gz' in workflow and "gates['deferred_numerics'] = command_gate" in runner and "'bench/deferred_numerical_experiments.py', '--check'" in runner,'CI queue check missing');checks['CI_global_runner_retains_producer_check']=True
queue=R/m.PLAIN;replay=P/'storage-replay/bench/tritonbench_g/deferred_numerical_experiments.json.gz';m.require(m.file_hash(queue)==m.decompressed_hash(replay),'storage replay no longer matches live plain queue');checks['live_plain_queue_bytes']=queue.stat().st_size;checks['live_plain_queue_sha256']=m.file_hash(queue)
raw=gzip.decompress((R/m.LEDGER).read_bytes());rebound=m.rebind_ledger_bytes(raw,m.file_hash(replay));oldledger=json.loads(raw);newledger=json.loads(rebound);m.require(oldledger['tasks']==newledger['tasks'],'task rows changed');m.require(newledger==m.rebind_ledger(oldledger,m.file_hash(replay)),'unexpected ledger field change');checks['real_current_ledger_in_memory_rebind_only']=True;checks['legacy_rows_preserved']=len(oldledger['tasks']);checks['legacy_task_rows_sha256']=m.digest(m.canonical(oldledger['tasks']));checks['ledger_decompressed_bytes_before']=len(raw);checks['ledger_decompressed_bytes_after']=len(rebound)
patches=[]
for rel in source['changes']:
 before=(R/rel).read_text().splitlines(True) if (R/rel).exists() else [];after=(W/'after'/rel).read_text().splitlines(True);patches.extend(difflib.unified_diff(before,after,fromfile='a/'+rel,tofile='b/'+rel))
(W/'evidence/candidate.patch').write_text(''.join(patches));m.write(W/'evidence/candidate-audit.json',{'status':'passed','checks':checks,'base_commit_claimed':None,'migration_executed':False,'repository_sources_modified':False,'primary_gate_modified':False,'new_dtype_profiles':0,'acceptance_increment':0});print(json.dumps(checks))
