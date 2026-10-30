import bench.tritonbench_g.embedding_triton_kernel.Proofs.Kernel
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.StructuralAccess
open VeriTile.Triton FP.Structural Execution
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Layout and active address obligations actually used by source safety.
Signed-word interpretation belongs to the public logical input domain. -/
structure AccessContract (p : Parameters) (pid : Nat) (ids : Nat → Int)
    (bounds : RegionBounds) : Prop extends Execution.Layout p where
  idAccess : ∀ a,a<64 → pid*64+a<p.n → pid*64+a<bounds p.ids
  weightAccess : ∀ a,a<64 → pid*64+a<p.n →
    p.first ≤ ids (pid*64+a) → ids (pid*64+a)<p.last → ∀ d,d<p.hidden →
    (ids (pid*64+a)-p.first).toNat*p.weightStride+d<bounds p.weight
  outputAccess : ∀ a,a<64 → pid*64+a<p.n → ∀ d,d<p.hidden →
    (pid*64+a)*p.outputStride+d<bounds p.output

def AccessContract.ofKernel (h : KernelContract p pid ids bounds) :
    AccessContract p pid ids bounds :=
  { ids_distinct := h.ids_distinct, weight_distinct := h.weight_distinct,
    width_covers := h.width_covers, stride_covers := h.stride_covers,
    idAccess := h.idAccess, weightAccess := h.weightAccess, outputAccess := h.outputAccess }

/-- One actual loop-body execution: metadata, gather and output accesses. -/
theorem chunk_safe (rest : Algebra α) (zero : α) (p : Parameters)
    (bounds : RegionBounds) (s : State α) (start : Nat) (hs : start<64)
    (context : Context p s)
    (h : AccessContract p (s.pids 0) (fun r => (s.mem p.ids r).read .int) bounds) :
    Safety.traceSafeList (backend rest zero) bounds (body .int32 p)
      (s.setReg "start_nn" .nat [] (fun _ => start)) := by
  rcases context with ⟨hr,hc⟩
  simp [body,source,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    copyBackend,backend,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,
    bop,ofFloat,toFloat,Region.cast,store,State.setReg,TileShape.dropInsertedIndex,hr,hc,
    Broadcast.leftIndex,Broadcast.rightIndex,Fin.eq_zero]
  constructor
  · intro hi
    simpa only [Nat.add_comm] using h.idAccess start hs (by simpa only [Nat.add_comm] using hi)
  constructor
  · intro i row lo hi col
    simp only [if_pos row] at lo hi
    simp only [row,lo,hi,and_self,if_true]
    rw [Execution.nonnegative_offset _ _ _ _ lo]
    simpa only [Nat.add_comm] using h.weightAccess start hs
      (by simpa only [Nat.add_comm] using row)
      (by simpa only [Nat.add_comm] using lo)
      (by simpa only [Nat.add_comm] using hi) i.val col
  · intro i row col
    simp only [if_pos row]
    simpa only [Nat.add_comm] using h.outputAccess start hs
      (by simpa only [Nat.add_comm] using row) i.val col

#axiomsClean chunk_safe
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.StructuralAccess
