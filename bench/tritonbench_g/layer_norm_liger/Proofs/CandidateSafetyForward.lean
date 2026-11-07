import bench.tritonbench_g.layer_norm_liger.Proofs.Support.KernelEntryAdapters
import VeriTile.Triton.Float.StructuralSafety
namespace VeriTile.Bench.Repairs.LayerNormLiger.CandidateSafety
open VeriTile.Triton FP.Structural Safety
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem forward (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters)
    (s : State α) (bounds : RegionName → Nat)
    (hx : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.xStride+i.val<bounds p.x)
    (hw : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.weight)
    (hb : ∀ i : Fin p.block,i.val<p.columns → i.val<bounds p.bias)
    (hm : s.pids 0*p.meanStride<bounds p.mean)
    (hr : s.pids 0*p.rstdStride<bounds p.rstd)
    (hy : ∀ i : Fin p.block,i.val<p.columns → s.pids 0*p.yStride+i.val<bounds p.y) :
    TraceSafe (withSum M R) bounds (Forward.candidateKernel p) s := by
  simp [TraceSafe,Forward.candidateKernel,Forward.candidateSourcePastedKernel,ComputeKernel.surfaceBody,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,
    ComputeDType.eraseDType,FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,
    numeric,bop,numericLt,natLt,Region.cast,withSum,hm,hr]
  simp only [←Int.natCast_mul,Int.toNat_natCast,hm,hr,true_and,and_true]
  exact ⟨hx,hw,hb,hy⟩

#axiomsClean forward
end
end VeriTile.Bench.Repairs.LayerNormLiger.CandidateSafety
