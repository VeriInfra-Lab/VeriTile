"""Check each generated case's first admission outcome against the Lean classifier."""
from pathlib import Path
import json,hashlib,subprocess,os,time,shutil
W=Path(__file__).resolve().parent;R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile');O=W/'evidence/case-admission';O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cases=json.loads((W/'evidence/scope-cases.json').read_text())
names={'bool':'boolean','uint8':'u8','int8':'i8','int16':'i16','int32':'i32','int64':'i64','uint16':'u16','uint32':'u32','uint64':'u64','fp16':'fp16','bf16':'bf16','fp32':'fp32','fp64':'fp64','complex32':'c32','complex64':'c64','complex128':'c128','fp8e4m3fn':'e4m3fn','fp8e5m2':'e5m2','fp8e4m3fnuz':'e4m3fnuz','fp8e5m2fnuz':'e5m2fnuz','fp8e8m0fnu':'e8m0fnu','packed_fp4e2m1':'e2m1x2'}
rejections={'unsupported_abs':('abs','unsupportedAbs'),'empty_maximum':('maximum','emptyMaximum'),'unsupported_maximum':('maximum','unsupportedMaximum'),'shape_unpack':('shapeUnpack','shapeUnpack'),'input_stride_assertion':('inputStrideAssertion','inputStrideAssertion'),'unsupported_signature':('signature','unsupportedSignature'),'unsupported_llrint_overload':('compilation','unsupportedLlrint'),'cpu_pointer':('pointerPacking','cpuPointer'),'grid_parse_limit':('gridParse','gridLimit 2147483648')}
checks={};mapping=[]
for c in cases:
 cfg=c['configuration'];d=names[c['dtype_profile']['inputs']['input']];shape=cfg['shape_class'];outcome=cfg['outcome'];site=cfg['first_primitive_error_stage']
 n,rank,M,N,stride={'empty':(0,2,0,3,True),'wrong_rank':(3,1,3,0,True),'bad_stride':(4,2,2,2,False),'valid_matrix':(3,2,3,1,True),'oversized_grid':(2**38,2,2**38,1,True)}[shape]
 env='(fun _ => none)' if site=='none' else '(fun s => if s=.'+site+' then some "probe" else none)'
 if outcome in ['returned_or_undefined_conversion','native_observation']:expected='.ok .'+d
 elif outcome=='primitive_error':expected='.error ⟨.'+site+',.primitive "probe"⟩'
 else:
  stage,reason=rejections[outcome];expected='.error ⟨.'+stage+',.'+reason+'⟩'
 expr=f'nativeAdmission {env} .{cfg["device"]} .{d} {n} {rank} Bool.{str(stride).lower()} {M} {N} = {expected}'
 checks.setdefault(expr,len(checks));mapping.append(checks[expr])
source='''import bench.tritonbench_g.quant_transpose_kernel.Proofs.FiniteNativeHost
namespace QuantTransposeFinite.Revision2Cases
open QuantTransposeFinite
set_option maxRecDepth 100000
set_option maxHeartbeats 200000000
noncomputable section
open Classical
def checks : List Bool := [\n'''+',\n'.join('  decide ('+c+')' for c in checks)+''']
theorem all_native_schema_cases : checks.all id=Bool.true := by decide +kernel
#axiomsClean all_native_schema_cases
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantTransposeFinite.Revision2Cases
'''
src=O/'AdmissionCases.lean';src.write_text(source);(O/'mapping.json').write_text(json.dumps({'scope_cases':len(cases),'unique_admission_profiles':len(checks),'case_to_profile':mapping},indent=2)+'\n')
base=subprocess.check_output(['/tmp/veritile-lean-toolchain/bin/lake','env','printenv','LEAN_PATH'],cwd=R,text=True).strip();env=dict(os.environ,LEAN_PATH=str(W)+':'+base)
cmd=['/tmp/veritile-lean-toolchain/bin/lean','--root='+str(W),str(src),'-o',str(O/'AdmissionCases.olean')];start=time.monotonic()
with (O/'build.log').open('w') as f:r=subprocess.run(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
if r.returncode==0:shutil.copyfile(O/'AdmissionCases.olean',O/'AdmissionCases.olean.bin')
result={'status':'passed' if r.returncode==0 else 'failed','exit_code':r.returncode,'scope_cases':len(cases),'unique_admission_profiles':len(checks),'seconds':time.monotonic()-start,'source_sha256':sha(src),'command':cmd,'LEAN_PATH':env['LEAN_PATH'],'scope_cases_sha256':sha(W/'evidence/scope-cases.json'),'checks':'Every generated case maps to a Lean-checked concrete first-stage representative, including all22 dtypes,CPU/CUDA,oversized grids and each reachable injected primitive failure; no runtime/GPU claim','artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file() and p.suffix!='.olean'}}
(O/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['artifacts','LEAN_PATH']},indent=2));raise SystemExit(r.returncode)
