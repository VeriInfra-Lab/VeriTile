"""Build immutable attempts with the existing lake environment; write only /tmp."""
from pathlib import Path
import datetime, hashlib, json, os, re, shutil, subprocess, sys, time

ROOT = Path(__file__).resolve().parent
REPO = Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
attempt = ROOT / 'builds' / stamp
src = attempt / 'src'
shutil.copytree(ROOT / 'src', src)
env = dict(os.environ)
paths = [str(Path(p).resolve()) for p in env.get('LEAN_PATH', '').split(':') if p]
assert any((Path(p) / 'Mathlib.olean').exists() for p in paths), paths
env['LEAN_PATH'] = ':'.join([str(src), *paths])
env['LEAN_NUM_THREADS'] = '2'
modules = sys.argv[1:] or ['QuantizeIntegerStage.Semantics', 'QuantizeIntegerStage.Global', 'QuantizeIntegerStage.Transpose', 'QuantizeIntegerStage.SourceChecks']
receipt = {'kind':'isolated_quantize_integer_stage_build', 'started_utc':stamp,
    'launcher':'PATH=/tmp/veritile-lean-toolchain/bin:$PATH LAKE_HOME=/tmp/veritile-lean-toolchain lake env python /tmp/veritile-standardization-20261015/quantize-integer-stage-wip/build.py',
    'lean_path':env['LEAN_PATH'], 'requested_modules':modules, 'records':[],
    'source_sha256':{str(p.relative_to(src)):sha(p) for p in src.rglob('*.lean')},
    'new_original_program_acceptances':0, 'status':'running'}
rp = attempt / 'receipt.json'
def save(): rp.write_text(json.dumps(receipt, indent=2)+'\n')
save()
for module in modules:
    source = src / (module.replace('.', '/')+'.lean')
    artifact = source.with_suffix('.olean')
    command = ['/tmp/veritile-lean-toolchain/bin/lean', '-R', str(src), '-o', str(artifact), str(source)]
    dep = subprocess.run([command[0], '--deps', str(source)], cwd=REPO, env=env, text=True, capture_output=True)
    dependencies = {p:sha(p) for p in dep.stdout.splitlines() if Path(p).is_file()}
    (attempt/(module+'.deps.log')).write_text(dep.stdout+dep.stderr)
    started=time.monotonic()
    result=subprocess.run(command,cwd=REPO,env=env,text=True,capture_output=True)
    log=attempt/(module+'.log');log.write_text(result.stdout+result.stderr)
    record={'module':module,'command':command,'exit_code':result.returncode,
        'elapsed_seconds':round(time.monotonic()-started,3),'source_sha256':sha(source),
        'log':str(log),'log_sha256':sha(log),'olean_sha256':sha(artifact) if artifact.exists() else None,
        'dependency_olean_sha256':dependencies, 'dependency_hashes_stable':all(sha(p)==h for p,h in dependencies.items()),
        'dependencies_exit_code':dep.returncode}
    receipt['records'].append(record);save()
    print(json.dumps({k:record[k] for k in ['module','exit_code','elapsed_seconds']}),flush=True)
    print(result.stdout+result.stderr,flush=True)
    if result.returncode or not record['dependency_hashes_stable']:break
receipt['status']='passed' if len(receipt['records'])==len(modules) and all(r['exit_code']==0 and r['dependency_hashes_stable'] for r in receipt['records']) else 'failed'
receipt['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps({'status':receipt['status'],'receipt':str(rp),'sha256':sha(rp)}),flush=True)
sys.exit(receipt['status']!='passed')
