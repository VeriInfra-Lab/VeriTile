from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,shutil,subprocess,sys
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');T=Path('/tmp/veritile-standardization-20261015');P=T/'rotary-emb-host-grid-wip/frozen-v2';O=T/'rotary-root-validation'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
checked={}
def exact(p,h):
 p=Path(p);assert sha(p)==h,str(p);checked[str(p)]=h
exact(P/'receipt.json','f954cfbc0aa96093c7d2406b791daf48c7188e8b33f608dfbd12d31a1d9ed5eb')
r=load(P/'receipt.json');exact(P/r['dependencies']['file'],r['dependencies']['sha256'])
exact(r['source']['path'],r['source']['sha256']);exact(P/'original/rotary_emb.py',r['source']['sha256'])
for rec in [r['toolchain']['lean'],r['toolchain']['lean_toolchain_file'],r['toolchain']['lake_manifest']]:exact(rec['path'],rec['sha256'])
exact(P/r['readme']['path'],r['readme']['sha256']);exact(P/r['first_overlay_failure']['path'],r['first_overlay_failure']['sha256'])
for tr in r['transformations']:
 exact(tr['source'],tr['source_sha256']);exact(P/tr['target'],tr['target_sha256'])
 text=Path(tr['source']).read_text()
 for old,new in r['import_mapping'].items():text=re.sub(r'^import '+old+r'$', 'import '+new,text,flags=re.M)
 assert text==(P/tr['target']).read_text()
for b in r['builds']:
 assert b['exit_code']==0;exact(P/b['log'],b['log_sha256']);assert 'error:' not in (P/b['log']).read_text()
b=r['original_failed_attempt_replay'];assert b['exit_code']==1;exact(P/b['log'],b['log_sha256'])
deps=load(P/'dependencies.json');assert len(deps)==5573 and all(d['source'] for d in deps)
moduleNames=sorted({line.removeprefix('dependency ') for line in (P/'evidence/SupportInventory.log').read_text().splitlines() if line.startswith('dependency ')})
assert moduleNames==[d['module'] for d in deps]
for d in deps:
 for f in [d['source'],*d['artifacts']]:
  exact(f['path'],f['sha256']);assert Path(f['path']).stat().st_size==f['bytes']
# Run the unchanged wrapper probe in a new directory; frozen files remain intact.
assert not O.exists();(O/'evidence').mkdir(parents=True)
shutil.copyfile(P/'probe_host.py',O/'probe_host.py')
p=subprocess.run([sys.executable,str(O/'probe_host.py')],capture_output=True,text=True,cwd=R)
(O/'probe.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
original=load(P/'evidence/HostProbe-source-ast.json');again=load(O/'evidence/HostProbe-source-ast.json')
assert original['case_count']==63 and original['all_passed'];assert {k:v for k,v in original.items() if k!='command'}=={k:v for k,v in again.items() if k!='command'}
expected=(O/'HostProbe.lean').read_text().replace('import HostMetadata\n','import bench.tritonbench_g.rotary_emb.Proofs.HostMetadata\n')
assert expected==(P/'overlay/bench/tritonbench_g/rotary_emb/Proofs/HostProbe.lean').read_text()
public=(P/'overlay/bench/tritonbench_g/rotary_emb/HostCorrect.lean').read_text();assert public.count('theorem rotary_emb_host_grid_correctness\n')==1
assert r['numerical_assumptions']==[]
local={p.relative_to(P).as_posix():sha(p) for p in P.rglob('*') if p.is_file() and not p.is_symlink() and not any(q.is_symlink() for q in p.parents if q!=P) and p.suffix not in {'.olean','.ilean'} and '.olean.' not in p.name}
record={'schema':1,'kind':'rotary_frozen_root_mechanical_validation','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','receipt_sha256':sha(P/'receipt.json'),'dependency_manifest_sha256':sha(P/'dependencies.json'),'imported_modules':len(deps),'checked_files':len(checked),'checked_file_sha256_manifest':hashlib.sha256(json.dumps(checked,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'frozen_local_artifact_sha256':local,'mapped_modules':8,'fresh_probes':63,'probe_log_sha256':sha(O/'probe.log'),'numerical_registrations':0,'acceptance_increment':0,'integration_required':'Register public specification by a recorded theorem-keyword transformation and rebuild; preserve original frozen theorem and all producer evidence.','preservation_note':r['preservation_note']}
(O/'validation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k] for k in ['status','imported_modules','checked_files','mapped_modules','fresh_probes']}));print('validation_sha256',sha(O/'validation.json'))
