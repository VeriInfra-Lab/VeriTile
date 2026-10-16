import bench.tritonbench_g.add_value.Proofs.NativeArithmetic
namespace AddValueNative
open VeriTile.Bench.TritonBenchG.VectorAddition
open DTypeScope
set_option maxRecDepth 24000
inductive InputType where
  | admitted (storage : Storage)
  | fnuzE4 | fnuzE5 | complex32 | complex64 | complex128 | e8m0 | packedFP4
  deriving DecidableEq

def InputType.name : InputType → String
  | .admitted d => TypedHost.dtypeName (TypedHost.wideDType d)
  | .fnuzE4 => "float8_e4m3fnuz" | .fnuzE5 => "float8_e5m2fnuz"
  | .complex32 => "complex32" | .complex64 => "complex64" | .complex128 => "complex128"
  | .e8m0 => "float8_e8m0fnu" | .packedFP4 => "float4_e2m1fn_x2"

def inputTypes : List InputType :=
  [.admitted .boolean,.admitted (.signed .w8),.admitted (.unsigned .w8),
   .admitted (.signed .w16),.admitted (.unsigned .w16),.admitted (.signed .w32),.admitted (.unsigned .w32),
   .admitted (.signed .w64),.admitted (.unsigned .w64),.admitted .fp16,.admitted .bf16,.admitted .fp32,
   .admitted .fp64,.admitted .f8e4fn,.admitted .f8e5,.fnuzE4,.fnuzE5,
   .complex32,.complex64,.complex128,.e8m0,.packedFP4]

def parseType (name : String) : Option InputType := inputTypes.find? (fun k ↦ k.name==name)

theorem parse_name (k : InputType) : parseType k.name=some k := by
  cases k with
  | admitted d => cases d <;> (try cases ‹FP64TypedCore.Width›) <;> decide +kernel
  | _ => decide +kernel


inductive SetupReject where | binder | architecture
  deriving DecidableEq, Repr
def admission : InputType → Except SetupReject Storage
  | .admitted d => .ok d
  | .fnuzE4 | .fnuzE5 => .error .architecture
  | _ => .error .binder
def outcomeStage : Except SetupReject Storage → Nat
  | .ok _ => 2
  | .error .architecture => 1
  | .error .binder => 0
def stageCount (s : Nat) : Nat :=
  (inputTypes.filter (fun t => outcomeStage (admission t)==s)).length
theorem admission_counts : inputTypes.length=22 ∧ stageCount 0=5 ∧ stageCount 1=2 ∧ stageCount 2=15 := by decide +kernel
#axiomsClean parse_name
#axiomsClean admission_counts
#guard_msgs (drop info) in
#auditModuleAxioms
end AddValueNative
