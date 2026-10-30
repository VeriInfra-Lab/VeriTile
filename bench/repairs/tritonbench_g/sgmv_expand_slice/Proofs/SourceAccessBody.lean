import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceMemory
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceAccess
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

/-- Bounds for the actual live input/weight lanes of one reduction block. -/
structure BodyBounds (bounds : RegionName → Nat) (p : KernelArgs)
    (c : SourceBody.Coordinates) (block : Nat) : Prop where
  input : ∀ i : TileIndex [32,16],
    p.inner % 16 = 0 ∨ SourceBody.globalInner block i.2.1 < p.inner →
    SourceBody.aOffset p c block i < bounds p.input
  weight : ∀ i : TileIndex [16,32],
    p.inner % 16 = 0 ∨ SourceBody.globalInner block i.1 < p.inner →
    SourceBody.bOffset p c block i < bounds p.weight

theorem body_safe {α : Type} [Inhabited α] (M : Algebra α)
    (dot : SourceBody.DotImplementation α) (abi : SourceABI) (p : KernelArgs)
    (c : SourceBody.Coordinates) (s t : State α) (block : Nat)
    (acc : Values α .real [32,32]) (h : SourceBody.Ready p c s t acc)
    (bounds : RegionName → Nat) (hb : BodyBounds bounds p c block) :
    Safety.traceSafeList (SourceBody.install M dot) bounds (SourceParts.body abi p)
      (t.setReg "k" .int [] (fun _ => (block : Int))) := by
  have ha := hb.input
  have hw := hb.weight
  simp only [SourceBody.aOffset,SourceBody.bOffset,SourceBody.safeInner,
    SourceBody.globalInner] at ha hw
  rw [SourceParts.body_same_ABI]
  rcases abi with ⟨input,weight,output,starts,lengths,adapters,add⟩
  cases input <;> cases weight <;>
    cases he : (p.inner % 16 == 0) <;>
    simp (maxSteps := 1000000) [SourceParts.body,SourceParts.loopParts,
      SourceParts.active,SourceParts.afterTile,SourceParts.afterLength,SourceParts.thenBody,
      SourceParts.outer,Source.intRange?,kernel,sourceFace,sourceHalfSingle323232,
      sourceBfloatSingle323232,ScalarFormat.dtype,LowFormat.scalar,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,SourceBody.install,
      State.setReg,he,h.inner,h.xm,h.xk,h.wa,h.wn,h.wk,h.start,h.adapter,h.offk,
      h.rows,h.columns,h.acc,h.mem,SourceArithmetic.input_offset,
      SourceArithmetic.weight_offset,SourceArithmetic.inner_lt]
  all_goals
    simp at he
    simp only [he,false_or,true_or,forall_const] at ha hw
    constructor
    · intro r k
      have address := SourceArithmetic.input_offset c.start (c.rows r) p.inputRowStride
        block k.val p.inner p.inputInnerStride
      simp only [SourceArithmetic.inner_lt] at address
      simpa only [address] using ha (r,k,PUnit.unit)
    · intro k col
      have address := SourceArithmetic.weight_offset p.adapterStride c.adapter
        block k.val p.inner p.weightInnerStride (c.columns col) p.weightColumnStride
      simp only [SourceArithmetic.inner_lt] at address
      simpa only [address] using hw (k,col,PUnit.unit)

#axiomsClean body_safe
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceAccess
