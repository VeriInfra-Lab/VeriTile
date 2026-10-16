from pathlib import Path
import hashlib,json,os,subprocess,time
W=Path(__file__).resolve().parent;BASE=W/'evidence/recorded-admission';BASE.mkdir(exist_ok=True);O=BASE/f'attempt{len(list(BASE.glob("attempt*")))+1:04d}';O.mkdir()
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=W/load(W/'evidence/native-latest.json')['receipt'];rows=load(p)['records']
constructors={'bool':'.admitted .boolean','uint8':'.admitted (.unsigned .w8)','int8':'.admitted (.signed .w8)','int16':'.admitted (.signed .w16)','uint16':'.admitted (.unsigned .w16)','int32':'.admitted (.signed .w32)','uint32':'.admitted (.unsigned .w32)','int64':'.admitted (.signed .w64)','uint64':'.admitted (.unsigned .w64)','float16':'.admitted .fp16','bfloat16':'.admitted .bf16','float32':'.admitted .fp32','float64':'.admitted .fp64','float8_e4m3fn':'.admitted .f8e4fn','float8_e5m2':'.admitted .f8e5','float8_e4m3fnuz':'.fnuzE4','float8_e5m2fnuz':'.fnuzE5','complex32':'.complex32','complex64':'.complex64','complex128':'.complex128','float8_e8m0fnu':'.e8m0','float4_e2m1fn_x2':'.packedFP4'}
def stage(row):
 r=row['result']
 if r['status']=='compiled':return 2
 if 'not supported in this architecture' in r['exception']:return 1
 assert r['failure_stage']=='signature' and 'KeyError' in r['exception'];return 0
entries=',\n  '.join('('+constructors[r['dtype']]+','+str(stage(r))+')' for r in rows)
source='''import bench.tritonbench_g.add_value.Proofs.NativeAdmission
namespace AddValueNative
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
def recordedStages : List (InputType × Nat) := [
  '''+entries+''']
theorem recorded_native_admission_agrees :
  recordedStages.all (fun (t,s) => outcomeStage (admission t)==s)=Bool.true := by decide +kernel
#axiomsClean recorded_native_admission_agrees
end AddValueNative
'''
src=O/'Matrix.lean';src.write_text(source)
last=load(sorted((W/'evidence/lean-attempts').glob('*.json'))[-1]);env=dict(os.environ,LEAN_PATH=last['LEAN_PATH']);cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(O),str(src),'-o',str(O/'Matrix.olean')]
start=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,env=env);(O/'compile.log').write_text(r.stdout+r.stderr)
result={'status':'passed' if r.returncode==0 else 'failed','exact_dtype_profiles_checked':len(rows),'native_receipt_sha256':sha(p),'source_sha256':sha(src),'command':cmd,'exit_code':r.returncode,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(O/'compile.log'),'artifacts':{q.name:sha(q) for q in O.iterdir() if q.is_file() and q.name!='receipt.json'}}
(O/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');(BASE/'latest.json').write_text(json.dumps({'receipt':str((O/'receipt.json').relative_to(W)),'sha256':sha(O/'receipt.json')},indent=2)+'\n');print(r.stdout+r.stderr);print(json.dumps(result));raise SystemExit(r.returncode)
