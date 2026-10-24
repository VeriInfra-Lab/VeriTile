import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.HoistedOutput
import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.SourceSafety
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedAccess
open VeriTile Triton FP.Structural FP.ProfiledRewrite Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem entry_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) (hb : Access.Bounds p s bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (HoistedSource.parts p).before s := by
  simp [HoistedSource.parts,HoistedSource.kernel,TypedSource.hoistedHeadMask,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Option.map,Option.getD,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,
    boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,
    ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,
    natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  constructor
  · intro i j hm hd
    exact hb.query (i,j,()) ⟨hm,hd⟩
  · exact hb.queryScale

theorem body_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : HoistedSource.Context M p initial row a current) (activeRow : row<p.length)
    (bounds : RegionName → Nat) (hb : Access.Bounds p initial bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (HoistedSource.parts p).body
      (current.setReg "start_n" .nat [] (fun _ => row)) := by
  simp [HoistedSource.parts,HoistedSource.kernel,TypedSource.hoistedHeadMask,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Option.map,Option.getD,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,
    boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,
    ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,
    natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    ctx.headMask,ctx.memory,ctx.rows,ctx.cols,ctx.query,ctx.scale,ctx.keyPointer,ctx.valuePointer,
    ctx.scalePointer,ctx.outputPointer,ctx.maximum,ctx.normalizer,ctx.value]
  exact ⟨fun i j hn hd => hb.key row activeRow (i,j,()) ⟨hn,hd⟩,
    hb.keyScale row activeRow,fun i j hn hd => hb.value row activeRow (i,j,()) ⟨hn,hd⟩⟩

theorem post_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : HoistedSource.Context M p initial row a current)
    (bounds : RegionName → Nat) (hb : Access.Bounds p initial bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (HoistedSource.parts p).after current := by
  simp [HoistedSource.parts,HoistedSource.kernel,TypedSource.hoistedHeadMask,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Option.map,Option.getD,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,
    boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,
    ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,
    natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    ctx.headMask,ctx.memory,ctx.rows,ctx.outputPointer,ctx.normalizer,ctx.value]
  constructor
  · intro i j hm hd
    exact hb.output (i,j,()) ⟨hm,hd⟩
  · split <;> trivial

/-- Safety follows the original quotation, actual loop states and all active
accesses. The separate existing execution theorem supplies termination. -/
theorem source_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) (hb : Access.Bounds p s bounds) :
    TraceSafe (engine (withDot M dot)) bounds (HoistedSource.kernel p) s := by
  change Safety.traceSafeList _ bounds (HoistedSource.kernel p).surfaceBody s
  rw [HoistedSource.source_body,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨entry_safe M dot p s bounds hb,?_⟩
  intro t ran
  have eq := Option.some.inj ((HoistedSource.entry M dot p s).property.symm.trans ran)
  subst t
  rw [List.singleton_append,Safety.traceSafeList_cons]
  refine ⟨?_,?_⟩
  · simp only [Safety.stepSafe_unfold]
    change Safety.rangeSafe (engine (withDot M dot)) bounds "start_n" 0 p.length 64 (HoistedSource.parts p).body (HoistedSource.entry M dot p s).val
    apply Safety.rangeSafe_of_invariant (engine (withDot M dot)) bounds "start_n" p.length 64 (HoistedSource.parts p).body
      (fun row t => ∃ a,HoistedSource.Context M p s row a t)
    · rintro row hr t ⟨a,ctx⟩
      exact body_safe M dot p s t row a ctx hr bounds hb
    · rintro row hr t ⟨a,ctx⟩ u hu
      have eq := Option.some.inj ((HoistedSource.bodyExec M dot p s t row a ctx).property.symm.trans hu)
      subst u
      exact ⟨update M dot p s row a,HoistedSource.body_context M dot p s t row a ctx⟩
    · exact ⟨initialAcc M,HoistedSource.entry_context M dot p s⟩
  · intro u hu
    obtain ⟨v,hv,row,ctx⟩ := HoistedSource.loopExec M dot p s (HoistedSource.entry M dot p s).val (HoistedSource.entry_context M dot p s)
    simp only [step] at hu
    change range (engine (withDot M dot)) "start_n" 0 p.length 64 (HoistedSource.parts p).body (HoistedSource.entry M dot p s).val=some u at hu
    have eq := Option.some.inj (hv.symm.trans hu)
    subst u
    exact post_safe M dot p s v row _ ctx bounds hb

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedAccess
