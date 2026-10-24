import json
from pathlib import Path
root=Path(__file__).resolve().parent
rows=json.loads((root/'evidence/source-routes/receipt.json').read_text())['profiles']
def lean_d(s):
 if s=='bool':return '.boolean'
 if s.startswith('uint'):return '(.unsigned .w'+s[4:]+')'
 if s.startswith('int'):return '(.signed .w'+s[3:]+')'
 return {'float16':'.fp16','bfloat16':'.bf16','float32':'.fp32','float64':'.fp64','float8_e4m3fn':'.f8e4fn','float8_e5m2':'.f8e5'}[s]
profile_list=',\n  '.join('('+lean_d(r['x'])+', '+lean_d(r['y'])+')' for r in rows)
expected=',\n  '.join('['+', '.join(json.dumps(t) for t in r['input_routes']['x']+r['input_routes']['y']+['arith.addf:f64']+r['output_route'])+']' for r in rows)
code='''import bench.tritonbench_g.vector_addition.Proofs.FP64TypedCore

/-! All29 supported fp64 operand pairs from the frozen frontend probes.
The route labels describe the expression dependency graph in postorder, not a
load/instruction trace and not a formal source extraction/compiler theorem. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.FP64Profiles
open FP64TypedCore
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

 def storages : List Storage := [
  .boolean, .signed .w8, .unsigned .w8, .signed .w16, .unsigned .w16,
  .signed .w32, .unsigned .w32, .signed .w64, .unsigned .w64,
  .fp16, .bf16, .fp32, .fp64, .f8e4fn, .f8e5]

def profiles : List (Storage × Storage) := [
  PROFILE_LIST]

instance (x y : Storage) : Decidable (Supported x y) :=
  inferInstanceAs (Decidable (x = .fp64 ∨ y = .fp64))

theorem storages_complete (d : Storage) : d ∈ storages := by
  cases d <;> try decide
  all_goals rename_i w; cases w <;> decide

private theorem membership_checked :
    storages.all (fun x => storages.all (fun y =>
      decide ((x,y) ∈ profiles ↔ Supported x y))) = true := by
  decide +kernel

/-- Complete and exact within the 15 supported storage dtypes. The two fnuz
formats have no successful captured frontend route and are not hidden here. -/
theorem profiles_exact (x y : Storage) : (x,y) ∈ profiles ↔ Supported x y := by
  have hx := List.all_eq_true.mp membership_checked x (storages_complete x)
  have hy := List.all_eq_true.mp hx y (storages_complete y)
  exact of_decide_eq_true hy

theorem profiles_cardinality : profiles.length = 29 ∧ profiles.Nodup := by
  decide +kernel

def widthName : Width → String
  | .w8 => "8" | .w16 => "16" | .w32 => "32" | .w64 => "64"

def castName : Cast s t → String
  | .boolLoad => "boolLoad:i8!=0->i1"
  | .boolTo64 => "arith.uitofp:i1->f64"
  | .boolStore => "boolStore:preserve-i8-byte"
  | .signedTo64 w => "arith.sitofp:i" ++ widthName w ++ "->f64"
  | .unsignedTo64 w => "arith.uitofp:i" ++ widthName w ++ "->f64"
  | .halfTo32 => "arith.extf:f16->f32"
  | .bfloatTo32 => "arith.extf:bf16->f32"
  | .singleTo64 => "arith.extf:f32->f64"
  | .e4To64 => "tt.fp_to_fp:f8E4M3FN->f64"
  | .e5To64 => "tt.fp_to_fp:f8E5M2->f64"
  | .toSigned w => "arith.fptosi:f64->i" ++ widthName w
  | .toUnsigned w => "arith.fptoui:f64->i" ++ widthName w
  | .toHalf => "arith.truncf:f64->f16"
  | .toBFloat => "arith.truncf:f64->bf16"
  | .toSingle => "arith.truncf:f64->f32"
  | .toE4RTNE => "tt.fp_to_fp:f64->f8E4M3FN:rtne"
  | .toE5RTNE => "tt.fp_to_fp:f64->f8E5M2:rtne"

def dependencyRoute : Expr xD yD d → List String
  | .x | .y => []
  | .cast c e => dependencyRoute e ++ [castName c]
  | .add64 l r => dependencyRoute l ++ dependencyRoute r ++ ["arith.addf:f64"]

/-- Literal expected graph labels copied from the checked raw-SSA route audit.
This is an independently readable finite expectation, not a call to source. -/
def expectedRoutes : List (List String) := [
  EXPECTED]

theorem all_routes_match :
    profiles.map (fun p => dependencyRoute (source p.1 p.2)) = expectedRoutes := by
  decide +kernel

#specNonCircular expectedRoutes avoiding [source, dependencyRoute]
#axiomsClean storages_complete
#axiomsClean profiles_exact
#axiomsClean profiles_cardinality
#axiomsClean all_routes_match
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VectorAddition.FP64Profiles
'''.replace('PROFILE_LIST',profile_list).replace('EXPECTED',expected).replace('\n def storages','\ndef storages')
(root/'bench/tritonbench_g/vector_addition/Proofs/FP64Profiles.lean').write_text(code)
