import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LayerNormOps.SourceSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward11 (M : Algebra α) (p : ForwardParameters) (s : State α)
    (bounds : RegionName → Nat)
    (hx : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.xStride+i.val<bounds p.x)
    (hw : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.weight)
    (hb : p.hasBias=Bool.true → ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.bias)
    (hres : p.hasResidual=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      s.pids 0*p.residualStride+i.val<bounds p.residual)
    (hy : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.yStride+i.val<bounds p.y)
    (hresout : p.storeResidual=Bool.true → ∀ i : Fin p.block,i.val<p.columns →
      s.pids 0*p.residualOutStride+i.val<bounds p.residualOut)
    (hm : p.rms=Bool.false → s.pids 0<bounds p.mean)
    (hr : s.pids 0<bounds p.rstd)
    (hR : p.rms=Bool.true) (hA : p.hasResidual=Bool.true) :
    TraceSafe M bounds (forwardKernel p) s := by
  cases hS : p.storeResidual <;> cases hB : p.hasBias
  all_goals
    simp [TraceSafe,forwardKernel,forwardSource,ComputeKernel.surfaceBody,
      traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
      safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
      ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
      numeric,bop,numericLt,natLt,Region.cast,hR,hA,hS,hB,hx,hw,hb,hres,hy,hresout,hm,hr]
  all_goals repeat' first
    | apply And.intro
    | exact hx
    | exact hw
    | exact hy
    | exact hr
    | exact hb hB
    | exact hres hA
    | exact hresout hS
    | exact hm hR


#axiomsClean forward11
end
end VeriTile.Bench.Repairs.LayerNormOps.SourceSafety
