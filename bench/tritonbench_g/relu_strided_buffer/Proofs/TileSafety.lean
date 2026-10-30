import bench.tritonbench_g.relu_strided_buffer.Proofs.Source
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.Access
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer StructuralTile SignedView SourceSignedBranch
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Bounds are checked at actual signed-address loads and stores, after the
complete signed offset is converted to a cell address. Masked lanes impose no
backing requirement, including negative inactive pointers. -/
theorem tile_safe {α : Type} [Inhabited α] (M : Algebra α) (cmp : α → α → Bool)
    (hc : M.compareLt none .fp32=some cmp) (c : Parameters) (T : Nat) (s : State α)
    (ht : s.regs .nat [] "tile_id"=some (fun _ => T)) (bounds : RegionName → Nat)
    (input : ∀ i,i<c.n → (views c).inputView.address i<bounds c.input.region)
    (output : ∀ i,i<c.n → (views c).outputView.address i<bounds c.output.region) :
    Safety.traceSafeList M bounds (perTile c) s := by
  simp [perTile,oneTileKernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,stepAlg,evalExpr,evalOp_unfold,numeric,numericLt,natLt,FP.Structural.bop,
    store,hc,ht,FloatDType.comparableDType,Region.cast,ofFloat,toFloat]
  constructor
  · intro i active
    exact input (T*c.B+i.val) active
  · intro i active
    exact output (T*c.B+i.val) active

#axiomsClean tile_safe
end
end VeriTile.Bench.Repairs.ReluStridedBuffer.Access
