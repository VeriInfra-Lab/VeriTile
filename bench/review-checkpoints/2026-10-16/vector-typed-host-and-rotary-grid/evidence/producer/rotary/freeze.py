from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parent
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
FROZEN=WORK/'frozen-v2'
LEAN='/tmp/veritile-lean-toolchain/bin/lean'
ORDER=['HostMetadata','HostGeometry','HostExecution','HostCoverage','HostCorrect','HostWitness','HostProbe','SupportInventory']
MAPPING={name:('bench.tritonbench_g.rotary_emb.'+('HostCorrect' if name=='HostCorrect' else 'Proofs.'+name)) for name in ORDER}

def digest(path):
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576),b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def record(path):
    return {'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size}

def run(command, log, env):
    start=time.monotonic()
    with log.open('w') as stream:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    entry={'command':command,'exit_code':result.returncode,'seconds':round(time.monotonic()-start,3),
           'log':str(log.relative_to(FROZEN)),'log_sha256':digest(log)}
    print(json.dumps(entry),flush=True)
    return entry

if FROZEN.exists():
    raise RuntimeError('Freeze directory already exists; preserve it and choose a new destination.')
(FROZEN/'evidence').mkdir(parents=True)
shutil.copytree(WORK/'evidence',FROZEN/'producer-evidence')
base=json.loads((WORK/'evidence'/'HostCorrect.json').read_text())['LEAN_PATH'].split(':')[1:]
env=dict(os.environ,LEAN_PATH=str(FROZEN/'overlay')+':'+':'.join(base))
transforms=[]
for name in ORDER:
    source=WORK/(name+'.lean')
    target=FROZEN/'overlay'/Path(*MAPPING[name].split('.')).with_suffix('.lean')
    target.parent.mkdir(parents=True,exist_ok=True)
    original=source.read_text()
    mapped=original
    for old,new in MAPPING.items():
        mapped=re.sub(r'^import '+old+r'$', 'import '+new,mapped,flags=re.MULTILINE)
    target.write_text(mapped)
    transforms.append({'source':str(source),'source_sha256':digest(source),
                       'target':str(target.relative_to(FROZEN)),'target_sha256':digest(target),
                       'transformation':'Only exact top-level local import names are replaced according to import_mapping.'})

# Lean resolves a namespace root once. Preserve existing bench artifacts below
# that root while shadowing only the eight added modules in this overlay.
def overlay_support(source, target):
    target.mkdir(parents=True,exist_ok=True)
    for entry in source.iterdir():
        dest=target/entry.name
        if dest.exists():
            if entry.is_dir() and dest.is_dir(): overlay_support(entry,dest)
            continue
        dest.symlink_to(entry,target_is_directory=entry.is_dir())
overlay_support(ROOT/'.lake/build/lib/lean/bench',FROZEN/'overlay/bench')

builds=[]
for name in ORDER:
    path=FROZEN/'overlay'/Path(*MAPPING[name].split('.')).with_suffix('.lean')
    command=[LEAN,'--root='+str(FROZEN/'overlay'),str(path),'-o',str(path.with_suffix('.olean'))]
    builds.append(run(command,FROZEN/'evidence'/(name+'.log'),env))
    if builds[-1]['exit_code']:
        (FROZEN/'failed-builds.json').write_text(json.dumps(builds,indent=2)+'\n')
        sys.exit(1)

# Replay the recovered exact original failure separately, without replacing the
# successful candidate or presenting reconstructed logs as original evidence.
replay=FROZEN/'failed-replay'
replay.mkdir()
old=WORK/'evidence'/'attempts'/'HostExecution-b5f7e68bf288.lean.txt'
(replay/'HostExecution.lean').write_bytes(old.read_bytes())
replay_env=dict(os.environ,LEAN_PATH=str(WORK)+':'+':'.join(base))
failed=run([LEAN,'--root='+str(replay),str(replay/'HostExecution.lean')],
           FROZEN/'evidence'/'original-HostExecution-replay.log',replay_env)
assert failed['exit_code']==1

modules=sorted({line.removeprefix('dependency ') for line in
    (FROZEN/'evidence'/'SupportInventory.log').read_text().splitlines() if line.startswith('dependency ')})
binary_roots=[FROZEN/'overlay',*[Path(p) for p in base]]
package_root=Path('/tmp/veritile-tritonbench-lake/packages')
source_roots=[FROZEN/'overlay',ROOT,*[p for p in sorted(package_root.iterdir()) if p.is_dir()],
              Path('/tmp/veritile-lean-toolchain/src/lean')]
dependencies=[]
for module in modules:
    relative=Path(*module.split('.'))
    binary=next((r/relative.with_suffix('.olean') for r in binary_roots if (r/relative.with_suffix('.olean')).is_file()),None)
    source=next((r/relative.with_suffix('.lean') for r in source_roots if (r/relative.with_suffix('.lean')).is_file()),None)
    assert binary is not None,module
    artifacts=[record(binary)]
    for suffix in ['.private','.server']:
        extra=Path(str(binary)+suffix)
        if extra.is_file(): artifacts.append(record(extra))
    dependencies.append({'module':module,'source':record(source) if source else None,'artifacts':artifacts})
(FROZEN/'dependencies.json').write_text(json.dumps(dependencies,indent=2)+'\n')
for name in ['compile.py','probe_host.py','freeze.py']:
    shutil.copy2(WORK/name,FROZEN/name)
shutil.copy2(WORK/'evidence'/'HostProbe-source-ast.json',FROZEN/'evidence'/'HostProbe-source-ast.json')
source=ROOT/'bench/tritonbench_g/rotary_emb/rotary_emb.py'
(FROZEN/'original').mkdir()
shutil.copy2(source,FROZEN/'original'/'rotary_emb.py')
receipt={'status':'producer_candidate; independent review and repository/catalog integration pending',
 'claim':'Original full-source exact-real-erased host/grid extension; no rounded/native acceptance claim.',
 'source':record(source),'import_mapping':MAPPING,'transformations':transforms,
 'builds':builds,'original_failed_attempt_replay':failed,'LEAN_PATH':env['LEAN_PATH'],
 'dependencies':{'file':'dependencies.json','sha256':digest(FROZEN/'dependencies.json'),'count':len(dependencies)},
 'toolchain':{'lean':record(Path(LEAN)),'version':subprocess.check_output([LEAN,'--version'],text=True).strip(),
              'lean_toolchain_file':record(ROOT/'lean-toolchain'),'lake_manifest':record(ROOT/'lake-manifest.json')},
 'public_statement':'VeriTile.Bench.TritonBenchG.RotaryEmb.HostGrid.rotary_emb_host_grid_correctness',
 'public_statement_log':'evidence/HostCorrect.log',
 'axioms':'Only standard Lean base propext, Classical.choice, Quot.sound; checked public, geometry, execution, coverage and witness theorems.',
 'host_probes':{'count':63,'all_passed':True,'receipt':'evidence/HostProbe-source-ast.json'},
 'numerical_assumptions':[],
 'trusted_primitives':['Python int(width * float_factor), including multiplication, truncation and exceptions',
     'Tensor rank/shape/stride/dtype metadata and primitive launch/load/store behavior; no generic runtime implementation proof'],
 'proved_scope':['BH=4, BS=16, HEAD_Q-derived ceil grid and both warp branches',
     'All original source statements execute at each scheduled PID on actual preceding memory',
     'Initial-memory independent logical Q/K real formulas, trace safety, raw-cell frame',
     'K heads beyond 4*ceil(HEAD_Q/4), odd and partial-dimension tails unchanged under layout ownership',
     'Finite full-grid coverage, arbitrary duplicate-free ordered execution helper, nonempty four-program strided witness'],
 'remaining_scope':['Rounded typed grid execution and a valid numerical specification/analysis remain unproved; real equality is not registered as a deferred floating-point law.',
     'Native dtype combinations, arange shape/power-of-two constraints and device resource admission are not inferred from real-erased storage parameters.',
     'Natural nonnegative cell strides and nonnegative converted dimension exclude negative strides/dimensions and finite machine-index overflow; native admissibility is separate.',
     'Q/K and trigonometric output-region separation, complete tensor-layout injectivity, logical width and active source-address bounds are explicit ownership/domain premises; aliasing outside this domain is not covered.',
     'Sequential region-memory source execution is proved; byte-level/native concurrency/launch runtime and a composed flat-byte launch theorem are not proved.',
     'Cos strides are used for both trig buffers even if Sin metadata differs; trig access bounds remain caller/domain obligations.',
     'Independent review, catalog/schema2 registration and full acceptance remain pending.'],
 'preservation_note':'All later failed attempts preserved. Exact pre-existing HostExecution source recovered and replayed; initial raw error log had been overwritten before preservation instruction, disclosed in producer-evidence/attempts/preexisting-recovery.json.'}
(FROZEN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('FROZEN_RECEIPT_SHA256 '+digest(FROZEN/'receipt.json'),flush=True)
