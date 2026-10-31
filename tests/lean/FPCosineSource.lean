import bench.tritonbench_g.cosine_compute.Kernels
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.CosineCompute.SourceTypes
open VeriTile.Triton
open VeriTile.Bench.Repairs.CosineCompute
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

def selected (swapped : Bool) (n B : Nat) :=
  if swapped then indexCommutedKernel "a" "b" n B else kernel "a" "b" n B

def assignType? (code : List ComputeStmt) (name : RegName) : Option TileDType :=
  match code with
  | [] => none
  | .assign d _ out _ :: rest => if out = name then some d else assignType? rest name
  | _ :: rest => assignType? rest name

def storeType? (code : List ComputeStmt) := code.getLast?.bind fun st => match st with
  | .store d _ _ _ _ => some d
  | _ => none

theorem preserved_types (swapped : Bool) (n B : Nat) :
    let code := (selected swapped n B).surfaceBody
    assignType? code "offset" = some .nat ∧
    assignType? code "a_value" = some .real ∧
    assignType? code "b_value" = some .real ∧
    storeType? code = some .real := by
  cases swapped <;> exact ⟨rfl,rfl,rfl,rfl⟩

/-- Source omits other=0: MaskOpt.mask retains unspecified masked values.
The padding tag is unused for an ordinary region pointer. -/
theorem load_padding (swapped : Bool) (n B : Nat) :
    ((selected swapped n B).surfaceBody.drop 2).head? =
      some (.assign .real [B] "a_value" (.compute
        (.load .fp32 (.region (⟨"a"⟩ : Region .real) (.ref .nat [B] "offset"))
          (.mask (.ref .bool [B] "mask"))))) := by
  cases swapped <;> rfl

/-- The intrinsic receives the explicit FP32 conversion. -/
theorem cosine_conversion (swapped : Bool) (n B : Nat) :
    ((selected swapped n B).surfaceBody.drop 3).head? =
      some (.assign .real [B] "b_value" (.compute (.alg .fp32
        (.cos (.castFloat .real .real (.ref .real [B] "a_value")))))) := by
  cases swapped <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CosineCompute.SourceTypes
