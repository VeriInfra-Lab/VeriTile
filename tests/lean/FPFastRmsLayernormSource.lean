import bench.tritonbench_g.fast_rms_layernorm.Kernels
import VeriTile.Meta.StatementAudit

/-! Precision regression for the actual repaired FP32 source and its current
integer-address candidate. Both constexpr Gemma branches and the runtime
scalar call binding are checked on the programs used by the public proofs. -/
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.SourceChecks
open VeriTile.Triton
open VeriTile.Bench.Repairs.FastRmsLayernorm.Kernels
set_option maxHeartbeats 1500000
set_option maxRecDepth 15000

 def types (body : List ComputeStmt) (name : RegName) : List TileDType :=
  body.filterMap fun st => match st with
    | .assign d _ n _ => if n=name then some d else none
    | _ => none

 def storeTypes (body : List ComputeStmt) : List TileDType :=
  body.filterMap fun st => match st with
    | .store d _ _ _ _ => some d
    | _ => none

 def regular (candidate : Bool) :=
  if candidate then regularRuntimeSourceIndexCommuted else regularRuntimeSource
 def gemma (candidate : Bool) :=
  if candidate then gemmaRuntimeSourceIndexCommuted else gemmaRuntimeSource
 def backward (candidate : Bool) :=
  if candidate then inputGradientSourceIndexCommuted else inputGradientSource

variable (candidate isGemma : Bool) (Y X W r dW : RegionName)
    (YR XR WR RR DWR N B : Nat) (eps : ℝ)

/-- The explicit cast into the selected weight storage is retained. -/
theorem regular_precision :
    (types (regular candidate Y X W r YR XR WR RR N B).surfaceBody "X_row",
     types (regular candidate Y X W r YR XR WR RR N B).surfaceBody "W_row",
     types (regular candidate Y X W r YR XR WR RR N B).surfaceBody "normed",
     types (regular candidate Y X W r YR XR WR RR N B).surfaceBody "output") =
      ([.real],[.real],[.real,.real],[.real]) := by
  cases candidate <;> rfl

theorem gemma_precision :
    (types (gemma candidate Y X W r YR XR WR RR N B).surfaceBody "W_row",
     types (gemma candidate Y X W r YR XR WR RR N B).surfaceBody "output") =
      ([.real],[.real]) := by
  cases candidate <;> rfl

theorem stores :
    storeTypes (regular candidate Y X W r YR XR WR RR N B).surfaceBody = [.real,.real] ∧
    storeTypes (gemma candidate Y X W r YR XR WR RR N B).surfaceBody = [.real,.real] ∧
    storeTypes (backward candidate Y X W r dW YR XR WR RR DWR N B eps isGemma).surfaceBody = [.real] := by
  cases candidate <;> cases isGemma <;> exact ⟨rfl,rfl,rfl⟩

/-- Binding epsilon adds precisely the call prelude, preserving every JIT statement. -/
theorem runtime_call :
    (regularSource Y X W r YR XR WR RR N B eps).surfaceBody =
      ScalarArgument.statement (ScalarArgument.fp32 "eps" eps) ::
        (regularRuntimeSource Y X W r YR XR WR RR N B).surfaceBody := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.SourceChecks
