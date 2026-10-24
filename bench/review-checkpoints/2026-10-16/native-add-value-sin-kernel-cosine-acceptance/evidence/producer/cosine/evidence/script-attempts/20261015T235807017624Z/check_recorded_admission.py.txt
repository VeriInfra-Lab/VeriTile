from pathlib import Path
import hashlib,json,os,subprocess,time
W=Path(__file__).resolve().parent;BASE=W/'evidence/recorded-admission';BASE.mkdir(exist_ok=True);O=BASE/f'attempt{len(list(BASE.glob("attempt*")))+1:04d}';O.mkdir()
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=W/load(W/'evidence/native-latest.json')['receipt'];rows=[dict(r['result'], N=4096) for r in load(p)['records']]
receipts=[p]
for q in sorted((W/'extent-frontend-attempts').glob('*/receipt.json')):
 receipts.append(q);rows+=load(q)['rows']
constructors={'bool':'.admitted .boolean','uint8':'.admitted (.unsigned .w8)','int8':'.admitted (.signed .w8)','int16':'.admitted (.signed .w16)','uint16':'.admitted (.unsigned .w16)','int32':'.admitted (.signed .w32)','uint32':'.admitted (.unsigned .w32)','int64':'.admitted (.signed .w64)','uint64':'.admitted (.unsigned .w64)','float16':'.admitted .fp16','bfloat16':'.admitted .bf16','float32':'.admitted .fp32','float64':'.admitted .fp64','float8_e4m3fn':'.admitted .f8e4fn','float8_e5m2':'.admitted .f8e5','float8_e4m3fnuz':'.fnuzE4','float8_e5m2fnuz':'.fnuzE5','complex32':'.complex32','complex64':'.complex64','complex128':'.complex128','float8_e8m0fnu':'.e8m0','float4_e2m1fn_x2':'.packedFP4'}
def stage(row):
 if row['status'] in ['compiled','frontend_passed']:return 4
 err=row.get('exception',row.get('error',''))
 if 'not supported in this architecture' in err:return 1
 if 'out of range for type int32' in err:return 2
 if 'TRITON_MAX_TENSOR_NUMEL' in err or 'maximum allowed' in err:return 3
 assert 'KeyError' in err,(row['dtype'],row['N'],err);return 0
entries=',\n  '.join('('+constructors[r['dtype']]+','+str(r['N'])+','+str(stage(r))+')' for r in rows)
source="""import bench.tritonbench_g.cosine_compute.Proofs.NativeAdmission
namespace CosineNative
set_option maxHeartbeats 120000000
set_option maxRecDepth 24000
def recordedStages : List (InputType × Nat × Nat) := [
  """+entries+"""]
theorem recorded_native_admission_agrees :
  recordedStages.all (fun (t,n,s) => outcomeStage (admission t n)==s)=Bool.true := by decide +kernel
#axiomsClean recorded_native_admission_agrees
end CosineNative
"""
src=O/'Matrix.lean';src.write_text(source)
last=load(sorted((W/'evidence/lean-attempts').glob('*.json'))[-1]);env=dict(os.environ,LEAN_PATH=last['LEAN_PATH']);cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(O),str(src),'-o',str(O/'Matrix.olean')]
start=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,env=env);(O/'compile.log').write_text(r.stdout+r.stderr)
result={'status':'passed' if r.returncode==0 else 'failed','exact_dtype_profiles_checked':len(rows),'native_receipts':{str(q.relative_to(W)):sha(q) for q in receipts},'source_sha256':sha(src),'command':cmd,'exit_code':r.returncode,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(O/'compile.log'),'artifacts':{q.name:sha(q) for q in O.iterdir() if q.is_file() and q.name!='receipt.json'}}
(O/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');(BASE/'latest.json').write_text(json.dumps({'receipt':str((O/'receipt.json').relative_to(W)),'sha256':sha(O/'receipt.json')},indent=2)+'\n');print(r.stdout+r.stderr);print(json.dumps(result));raise SystemExit(r.returncode)
