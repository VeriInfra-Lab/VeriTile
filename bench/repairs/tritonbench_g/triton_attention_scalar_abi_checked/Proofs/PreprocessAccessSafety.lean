import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.AccessGeometry

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem preprocess_safe (rest : Algebra α) (P : Primitives α)
    (p : PreParameters) (s : State α) (bounds : RegionName→Nat)
    (output : ∀ i : PreLane p,preOffset p s i<bounds p.output)
    (gradient : ∀ i : PreLane p,preOffset p s i<bounds p.gradient)
    (denominator : ∀ i : PreRow p,preRow p s i<bounds p.denominator)
    (newGradient : ∀ i : PreLane p,preOffset p s i<bounds p.newGradient)
    (delta : ∀ i : PreRow p,preRow p s i<bounds p.delta) :
    TraceSafe (backend rest P) bounds (preSource p) s := by
  simp [TraceSafe,preSource,preprocessTyped,ComputeKernel.surfaceBody,
    AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,store,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,backend,
    numeric,ComputeDType.eraseDType,ofFloat,toFloat,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,State.setReg,State.scatter_regs]
  exact ⟨fun i d=>output (i,d,PUnit.unit),fun i d=>gradient (i,d,PUnit.unit),
    fun i=>denominator (i,PUnit.unit),fun i d=>newGradient (i,d,PUnit.unit),fun i=>delta (i,PUnit.unit)⟩

#axiomsClean preprocess_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
