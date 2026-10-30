import bench.tritonbench_g.softmax_triton3.Kernels
import bench.tritonbench_g.softmax_triton2.Proofs.GridPair

namespace VeriTile.Bench.Repairs.SoftmaxTriton3.Pair
open VeriTile.Triton FP.Structural FP.Guarded
open VeriTile.Bench.Repairs.SoftmaxTriton2.GuardedReciprocal
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def numeratorValue {α : Type} [Inhabited α] (M : Algebra α)
    (inp mask : RegionName) (stride n B : Nat) (hasMask : Bool) (s : State α) : Values α .real [B] :=
  let row : Values α .real [B] := fun j => M.cast (some .fp32) .real .real
    (if j.1.val < n then (s.mem inp (s.pids 0*stride+j.1.val)).read .real else M.negInf)
  let maximum := M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false row PUnit.unit
  let shifted := fun j => M.binary (some .fp32) .real .sub (row j) maximum
  let shifted := if hasMask then fun j => M.binary (some .fp32) .real .add (shifted j)
    (M.cast (some .fp32) .real .real (if j.1.val < n then
      (s.mem mask (s.pids 0*stride+j.1.val)).read .real else M.literal (some .fp32) .real 0))
    else shifted
  fun j => M.unary (some .fp32) .exp (shifted j)

def denominatorValue {α : Type} [Inhabited α] (M : Algebra α)
    (inp mask : RegionName) (stride n B : Nat) (hasMask : Bool) (s : State α) : α :=
  M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
    (numeratorValue M inp mask stride n B hasMask s) PUnit.unit

def originalExecution {α : Type} [Inhabited α] (M : Algebra α)
    (out inp mask : RegionName) (stride n B : Nat) (hasMask : Bool)
    (hB : 0 < B) (s : State α) :
    {t : State α // exec (FP.ProfiledRewrite.engine M)
      (softmax_kernel out inp mask stride n B hasMask) s = some t} := by
  cases hasMask
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [FP.Structural.exec,softmax_kernel,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,store,State.setReg,numeric,natLt,bop,
      TileShape.axisDim,hB,Option.bind,Function.comp_def,ComputeDType.eraseDType]
    rfl

/-- Every mask selection and every runtime fast/fallback guard case succeeds.
Explicit FP32 casts and their positions are retained in both sources. -/
def candidateExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (ports : Ports M D) (out inp mask : RegionName)
    (stride n B : Nat) (hasMask : Bool) (hB : 0 < B) (s : State α) :
    {t : State α // exec (FP.ProfiledRewrite.engine M)
      (guarded_reciprocal_kernel out inp mask stride n B hasMask) s = some t} := by
  classical
  let num := numeratorValue M inp mask stride n B hasMask s
  let den := denominatorValue M inp mask stride n B hasMask s
  by_cases hc : TileShape.discreteBoolSum ⟨0,by simp⟩ Bool.false
      (fun i => ports.finite (num i)) PUnit.unit = B
  all_goals by_cases hf : ports.finite den = Bool.true
  all_goals by_cases hz : ports.atLeast (M.literal (some .fp32) .real 1) den = Bool.true
  all_goals cases hasMask
  all_goals
    refine ⟨?_,?_⟩
    swap
    dsimp [num,den,numeratorValue,denominatorValue] at hc hf hz
    simp [FP.Structural.exec,guarded_reciprocal_kernel,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,store,State.setReg,numeric,natLt,bop,
      TileShape.axisDim,hB,Option.bind,Function.comp_def,ComputeDType.eraseDType,
      ports.finite_bind,ports.order_bind,numericEq,numericLe,FP.ProfiledRewrite.engine,
      Algebra.withDefaultPrecision,resolvePrecision,ofFloat,toFloat,hc,hf,hz]
    rfl

#axiomsClean originalExecution
#axiomsClean candidateExecution
end
end VeriTile.Bench.Repairs.SoftmaxTriton3.Pair
