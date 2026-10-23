import bench.tritonbench_g.index_select_bwd.FPEquiv

/-! Historical overwrite-source candidate. This file does not certify the
accumulation repair: both old variants omit the destination add. -/
namespace VeriTile.Bench.Repairs.IndexSelectBwd.OriginalRewrite
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.IndexSelectBwd
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

/-- The exact historical FP32/FP32/int64 source pair, differing only in the
column product pid1*BC versus BC*pid1. -/
abbrev original := source .fp32 .fp32 .int64 Bool.false
abbrev candidate := source .fp32 .fp32 .int64 Bool.true

theorem execution_eq {A : Type} [Inhabited A] (M : Algebra A)
    (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) (s : State A) :
    exec M (original out idx inp rows n c s0 s1 BI BC) s=
      exec M (candidate out idx inp rows n c s0 s1 BI BC) s := by
  have firstSteps : run M ((original out idx inp rows n c s0 s1 BI BC).surfaceBody.take 3) s=
      run M ((candidate out idx inp rows n c s0 s1 BI BC).surfaceBody.take 3) s := by
    simp [original,candidate,source,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalOp_unfold,numeric,ComputeDType.eraseDType,Option.bind]
    congr 1
    funext i
    simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
  change run M (original out idx inp rows n c s0 s1 BI BC).surfaceBody s=
    run M (candidate out idx inp rows n c s0 s1 BI BC).surfaceBody s
  rw [←List.take_append_drop 3 (original out idx inp rows n c s0 s1 BI BC).surfaceBody,
    ←List.take_append_drop 3 (candidate out idx inp rows n c s0 s1 BI BC).surfaceBody]
  rw [run_append,run_append,firstSteps]
  rfl

/-- Constructive success of the entire old source, including masked signed
indices and the plain store. No assumption of a successful run is required. -/
def execution {A : Type} [Inhabited A] (M : Algebra A)
    (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) (s : State A) :
    {t : State A // exec M (original out idx inp rows n c s0 s1 BI BC) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [original,source,FP.Structural.exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
    Option.bind,numericLt,natLt,bop,numeric,store,Region.cast,
    ComputeDType.eraseDType,ofFloat,toFloat]
  rfl

/-- Both overwrite variants succeed and agree on complete memory. This is
only an integer-address rewrite, not equivalence with the backward formula. -/
theorem successful_equivalence {A : Type} [Inhabited A] (M : Algebra A)
    (out idx inp : RegionName) (rows n c s0 s1 BI BC : Nat) (s : State A) :
    ∃ left right,
      exec M (original out idx inp rows n c s0 s1 BI BC) s=some left ∧
      exec M (candidate out idx inp rows n c s0 s1 BI BC) s=some right ∧
      left.mem=right.mem := by
  let t := execution M out idx inp rows n c s0 s1 BI BC s
  exact ⟨t.val,t.val,t.property,(execution_eq M out idx inp rows n c s0 s1 BI BC s).symm.trans t.property,rfl⟩

#axiomsClean execution_eq
#axiomsClean execution
#axiomsClean successful_equivalence
end
end VeriTile.Bench.Repairs.IndexSelectBwd.OriginalRewrite
