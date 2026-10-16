from pathlib import Path
import json,subprocess,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;nr=json.loads((W/'evidence/native-latest.json').read_text());rows=json.loads((W/nr['receipt']).with_name('records.json').read_text())
items=[]
for row in rows:
 stage=3 if row['status']=='compiled' else 0 if row['phase']=='binder' else 1 if 'fnuz' in row['dtype'] else 2
 items.append('('+json.dumps(row['dtype'])+','+str(stage)+')')
s='''import bench.tritonbench_g.sin_kernel.Proofs.NativeAdmission
namespace SinKernelNative
theorem recorded_original_admission :
  (inputTypes.map (fun k => (k.name,outcomeStage (admission k)))) =
  ['''+','.join(items)+'''] := by decide +kernel
#axiomsClean recorded_original_admission
#guard_msgs (drop info) in
#auditModuleAxioms
end SinKernelNative
'''
(W/'after/RecordedSineAdmission.lean').write_text(s)
r=subprocess.run([sys.executable,str(W/'compile.py'),'RecordedSineAdmission.lean'],capture_output=True,text=True);print(r.stdout);raise SystemExit(r.returncode)
