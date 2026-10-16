from pathlib import Path
import hashlib,json,os,subprocess,time
W=Path(__file__).resolve().parent;BASE=W/'recorded-admission';BASE.mkdir(exist_ok=True);O=BASE/f'attempt{len(list(BASE.glob("attempt*")))+1:04d}';O.mkdir()
load=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=W/'replay'/load(W/'replay/evidence/native-latest.json')['receipt'];rows=load(p.with_name('records.json'))
canonical={'bool':'b1','int8':'i8','uint8':'u8','int16':'i16','uint16':'u16','int32':'i32','uint32':'u32','int64':'i64','uint64':'u64','float16':'fp16','bfloat16':'bf16','float32':'fp32','float64':'fp64','float8_e4m3fn':'fp8e4nv','float8_e5m2':'fp8e5','float8_e4m3fnuz':'fp8e4b8','float8_e5m2fnuz':'fp8e5b16'}
extra={'complex32':'complex32','complex64':'complex64','complex128':'complex128','float8_e8m0fnu':'e8m0','float4_e2m1fn_x2':'fp4packed'}
def ty(name):return '(.canonical .'+canonical[name]+')' if name in canonical else '.'+extra[name]
def observed(r):
 if r.get('backend')=='compiled':return 4
 if r.get('backend')=='rejected':return 3
 if r['error_type']=='KeyError':return 0
 if 'not supported in this architecture' in r['error']:return 1
 assert 'unexpected type' in r['error'],r
 return 2
pairs=',\n  '.join('('+ty(r['x'])+','+ty(r['y'])+','+str(observed(r))+')' for r in rows)
source='''import bench.tritonbench_g.vector_addition_custom.Proofs.NativeAdmission
namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
def recordedNativeStages : List (DenseDType × DenseDType × Nat) := [
  '''+pairs+''']
theorem recorded_native_matrix_agrees :
  recordedNativeStages.all (fun (x,y,stage) => outcomeStage (admission x y)==stage)=Bool.true := by decide +kernel
#axiomsClean recorded_native_matrix_agrees
end VeriTile.Bench.TritonBenchG.VectorAdditionCustom.Native
'''
src=O/'Matrix.lean';src.write_text(source)
last=load(W/'independent-build-receipt.json')['modules'][0];env=dict(os.environ,LEAN_PATH=last['LEAN_PATH']);cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(O),str(src),'-o',str(O/'Matrix.olean')]
start=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,env=env);(O/'compile.log').write_text(r.stdout+r.stderr)
result={'status':'passed' if r.returncode==0 else 'failed','exact_pairs_checked':len(rows),'native_records_sha256':sha(p.with_name('records.json')),'native_receipt_sha256':sha(p),'source_sha256':sha(src),'command':cmd,'exit_code':r.returncode,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(O/'compile.log'),'artifacts':{q.name:sha(q) for q in O.iterdir() if q.is_file() and q.name!='receipt.json'}}
(O/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');(BASE/'latest.json').write_text(json.dumps({'receipt':str((O/'receipt.json').relative_to(W)),'sha256':sha(O/'receipt.json')},indent=2)+'\n');print(r.stdout+r.stderr);print(json.dumps(result));raise SystemExit(r.returncode)
