import concurrent.futures,gzip,hashlib,json,subprocess,time
from pathlib import Path
root=Path.cwd()
out=Path('bench/review-checkpoints/2026-10-23/selected-source-replay')
out.mkdir(parents=True,exist_ok=True)
checks=out/'checks';checks.mkdir(exist_ok=True)
ledger=json.loads(Path('bench/tritonbench_g/task_progress.json').read_text())
formulas=json.loads(Path('bench/review-checkpoints/2026-10-23/reference-separation/main-formulas.json').read_text())
heads={r['task']:r['current_main_declarations'] for r in formulas['tasks']}
targets={t['task']:[str(Path(p).with_suffix('')).replace('/','.') for p in t['kernel_revision']['public_entries']] for t in ledger['tasks']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(task):
    modules=targets[task]
    path=checks/(task+'.lean')
    path.write_text('\n'.join(['import VeriTile.Meta.StatementAudit',*['import '+m for m in modules],'',*['#axiomsClean '+n for n in heads[task]]])+'\n')
    command=['lake','env','lean',str(path)]
    started=time.monotonic()
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log=checks/(task+'.log.gz');log.write_bytes(gzip.compress(result.stdout,mtime=0))
    record={'task':task,'modules':modules,'declarations':heads[task],'command':command,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-started,'check_source':{'path':str(path),'sha256':sha(path)},'log':{'path':str(log),'sha256':sha(log)},'raw_log_sha256':hashlib.sha256(result.stdout).hexdigest()}
    print(task,result.returncode,flush=True)
    return record
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(check,targets))
(out/'joint-checks.json').write_text(json.dumps({'schema':1,'tasks':results,'task_count':len(results),'passed':sum(r['exit_code']==0 for r in results),'failed':sum(r['exit_code']!=0 for r in results)},indent=2)+'\n')

# Exercise the shared namespace and import extensions in one Lean environment.
all_source=out/'all-selected.lean'
all_source.write_text('\n'.join(['import VeriTile.Meta.StatementAudit',*['import '+m for m in sorted({m for ms in targets.values() for m in ms})],'',*['#axiomsClean '+n for names in heads.values() for n in names]])+'\n')
started=time.monotonic()
command=['lake','env','lean',str(all_source)]
run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
all_log=out/'all-selected.log.gz';all_log.write_bytes(gzip.compress(run.stdout,mtime=0))
(out/'all-selected.json').write_text(json.dumps({'command':command,'exit_code':run.returncode,'elapsed_seconds':time.monotonic()-started,'source':{'path':str(all_source),'sha256':sha(all_source)},'log':{'path':str(all_log),'sha256':sha(all_log)},'raw_log_sha256':hashlib.sha256(run.stdout).hexdigest()},indent=2)+'\n')
print('combined',run.returncode,flush=True)
