import bench.tritonbench_g.cosine_compute.Proofs.NativeArithmetic
namespace CosineNative
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


inductive SetupReject where | binder | architecture | blockScalar | tensorShape
  deriving DecidableEq, Repr
/-- Signature types are formed before the body. For admitted pointer types,
the constexpr BLOCK is cast into the i32 multiply before arange checks its
power-of-two tensor size. Zero extent never reaches this stage. -/
def admission (kind : InputType) (n : Nat) : Except SetupReject Storage :=
  match kind with
  | .admitted d =>
    if HostMath.block n≥2^31 then .error .blockScalar
    else if HostMath.block n>2^20 then .error .tensorShape
    else .ok d
  | .fnuzE4 | .fnuzE5 => .error .architecture
  | _ => .error .binder
def outcomeStage : Except SetupReject Storage → Nat
  | .error .binder => 0
  | .error .architecture => 1
  | .error .blockScalar => 2
  | .error .tensorShape => 3
  | .ok _ => 4
def stageCount (n s : Nat) : Nat :=
  (inputTypes.filter (fun t => outcomeStage (admission t n)==s)).length

theorem admission_counts : inputTypes.length=22 ∧ stageCount 4096 0=5 ∧
    stageCount 4096 1=2 ∧ stageCount 4096 4=15 := by decide +kernel
theorem shape_failure_counts : stageCount (2^40+1) 0=5 ∧
    stageCount (2^40+1) 1=2 ∧ stageCount (2^40+1) 3=15 := by decide +kernel
theorem scalar_failure_counts : stageCount (2^61) 0=5 ∧
    stageCount (2^61) 1=2 ∧ stageCount (2^61) 2=15 := by decide +kernel
#axiomsClean shape_failure_counts
#axiomsClean scalar_failure_counts
#axiomsClean parse_name
#axiomsClean admission_counts
#guard_msgs (drop info) in
#auditModuleAxioms
end CosineNative
