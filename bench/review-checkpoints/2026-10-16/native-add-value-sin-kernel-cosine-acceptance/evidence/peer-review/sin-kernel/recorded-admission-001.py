from pathlib import Path
import hashlib,json,os,subprocess,time
W=Path(__file__).resolve().parent;BASE=W/'recorded-admission';BASE.mkdir(exist_ok=True);O=BASE/f'attempt{len(list(BASE.glob("attempt*")))+1:04d}';O.mkdir()
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=W/'replay'/load(W/'replay/evidence/native-latest.json')['receipt'];rows=load(p.with_name('records.json'))
entries=[]
for row in rows:
 stage=3 if row['status']=='compiled' else 0 if row['phase']=='binder' else 1 if 'not supported in this architecture' in row.get('error','') else 2
 entries.append('('+json.dumps(row['dtype'])+','+str(stage)+')')
source='import bench.tritonbench_g.sin_kernel.Proofs.NativeAdmission\nnamespace SinKernelNative\ntheorem recorded_original_admission :\n (inputTypes.map (fun k => (k.name,outcomeStage (admission k)))) = ['+','.join(entries)+'] := by decide +kernel\n#axiomsClean recorded_original_admission\nend SinKernelNative\n'
src=O/'Matrix.lean';src.write_text(source)
last=load(W/'independent-build-receipt.json')['modules'][0];env=dict(os.environ,LEAN_PATH=last['LEAN_PATH']);cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(O),str(src),'-o',str(O/'Matrix.olean')]
start=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,env=env);(O/'compile.log').write_text(r.stdout+r.stderr)
result={'status':'passed' if r.returncode==0 else 'failed','exact_dtype_profiles_checked':len(rows),'native_receipt_sha256':sha(p),'source_sha256':sha(src),'command':cmd,'exit_code':r.returncode,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(O/'compile.log'),'artifacts':{q.name:sha(q) for q in O.iterdir() if q.is_file() and q.name!='receipt.json'}}
(O/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');(BASE/'latest.json').write_text(json.dumps({'receipt':str((O/'receipt.json').relative_to(W)),'sha256':sha(O/'receipt.json')},indent=2)+'\n');print(r.stdout+r.stderr);print(json.dumps(result));raise SystemExit(r.returncode)
