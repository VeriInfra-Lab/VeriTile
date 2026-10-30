import bench.tritonbench_g.sgmv_expand_slice.Proofs.SourceSuffix
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceAccess
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

theorem suffix_safe {α : Type} [Inhabited α] (M : Algebra α)
    (abi : SourceABI) (hw : abi.weight=.half) (ho : abi.output=.single)
    (p : KernelArgs) (c : SourceSuffix.Coordinates) (s t : State α)
    (acc : Values α .real [32,32]) (h : SourceSuffix.Ready p c s t acc)
    (hlength : (s.mem p.lengths p.batch).read .int = (c.length : Int))
    (bounds : RegionName → Nat) (metadata : p.batch < bounds p.lengths)
    (output : ∀ i, SourceSuffix.active p c i → SourceSuffix.offset p c i < bounds p.output) :
    Safety.traceSafeList M bounds (SourceParts.suffix abi p) t := by
  have hout := output
  simp only [SourceSuffix.active,SourceSuffix.offset,SourceSuffix.row,SourceSuffix.column] at hout
  rw [SourceSuffix.common]
  rcases abi with ⟨input,weight,out,starts,lengths,adapters,add⟩
  dsimp only at hw ho
  subst weight
  subst out
  cases add <;>
    simp (maxSteps := 1000000) [SourceParts.suffix,SourceParts.active,SourceParts.afterTile,
      SourceParts.afterLength,SourceParts.thenBody,SourceParts.outer,kernel,sourceFace,
      sourceHalfSingle323232,ComputeKernel.surfaceBody,ScalarFormat.dtype,LowFormat.scalar,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      FP.Structural.run,FP.Structural.step,FP.Structural.store,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,bop,numericLt,Region.cast,ofFloat,toFloat,
      State.setReg,h.mem,h.undef,h.start,h.row,h.column,h.length,h.columns,h.batch,
      h.rowStride,h.columnStride,h.slice,h.acc,hlength,metadata,
      SourceStoreArithmetic.output_offset,SourceStoreArithmetic.row_lt,SourceStoreArithmetic.column_lt]
  all_goals
    repeat' apply And.intro
    all_goals
      intro r col hr hc
      have address := SourceStoreArithmetic.output_offset c.start r.val c.tileRow c.length
        col.val c.tileColumn p.slice p.columns p.outputRowStride p.outputColumnStride
      simp only [SourceStoreArithmetic.row_lt,SourceStoreArithmetic.column_lt] at address
      simpa only [address] using hout (r,col,PUnit.unit) ⟨hr,hc⟩

#axiomsClean suffix_safe
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceAccess
