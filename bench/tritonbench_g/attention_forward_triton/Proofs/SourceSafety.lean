import bench.tritonbench_g.attention_forward_triton.Proofs.Output
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Access
open VeriTile Triton FP.Structural FP.ProfiledRewrite Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Only the actual masked Q/K/V/output addresses and unmasked scale loads.
Pointer formation at an inactive lane does not demand an allocation. -/
structure Bounds (p : Parameters) (s : State α) (bounds : RegionName → Nat) : Prop where
  query : ∀ i : TileIndex [128,128], active p s i →
    base p s p.q+(s.pids 0*128+i.1.val)*p.q.row+i.2.1.val*p.q.dim < bounds p.Q
  queryScale : s.pids 1*((p.length+127)/128)+s.pids 0 < bounds p.QS
  key : ∀ row, row<p.length → ∀ i : TileIndex [128,64],
    i.2.1.val<p.length-row ∧ i.1.val<96 →
      base p s p.k+i.1.val*p.k.dim+(row+i.2.1.val)*p.k.row < bounds p.K
  keyScale : ∀ row, row<p.length →
    s.pids 1*((p.length+63)/64)+row/64 < bounds p.KS
  value : ∀ row, row<p.length → ∀ i : TileIndex [64,128],
    i.1.val<p.length-row ∧ i.2.1.val<96 →
      base p s p.v+(row+i.1.val)*p.v.row+i.2.1.val*p.v.dim < bounds p.V
  output : ∀ i : TileIndex [128,128], active p s i → address p s i < bounds p.Out

theorem entry_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (parts p).before s := by
  simp [parts,kernel,TypedSource.repaired,
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
    (ctx : Context M p initial row a current) (activeRow : row<p.length)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (parts p).body
      (current.setReg "start_n" .nat [] (fun _ => row)) := by
  simp [parts,kernel,TypedSource.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Option.map,Option.getD,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,
    boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,
    ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,
    natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    ctx.memory,ctx.rows,ctx.cols,ctx.query,ctx.scale,ctx.keyPointer,ctx.valuePointer,
    ctx.scalePointer,ctx.outputPointer,ctx.maximum,ctx.normalizer,ctx.value]
  exact ⟨fun i j hn hd => hb.key row activeRow (i,j,()) ⟨hn,hd⟩,
    hb.keyScale row activeRow,fun i j hn hd => hb.value row activeRow (i,j,()) ⟨hn,hd⟩⟩

theorem post_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    Safety.traceSafeList (engine (withDot M dot)) bounds (parts p).after current := by
  simp [parts,kernel,TypedSource.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Option.map,Option.getD,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,
    boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,
    ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,
    natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    ctx.memory,ctx.rows,ctx.outputPointer,ctx.normalizer,ctx.value]
  constructor
  · intro i j hm hd
    exact hb.output (i,j,()) ⟨hm,hd⟩
  · split <;> trivial

/-- Safety follows the original quotation, actual loop states and all active
accesses. The separate existing execution theorem supplies termination. -/
theorem source_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    TraceSafe (engine (withDot M dot)) bounds (kernel p) s := by
  change Safety.traceSafeList _ bounds (kernel p).surfaceBody s
  rw [source_body,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨entry_safe M dot p s bounds hb,?_⟩
  intro t ran
  have eq := Option.some.inj ((entry M dot p s).property.symm.trans ran)
  subst t
  rw [List.singleton_append,Safety.traceSafeList_cons]
  refine ⟨?_,?_⟩
  · simp only [Safety.stepSafe_unfold]
    change Safety.rangeSafe (engine (withDot M dot)) bounds "start_n" 0 p.length 64 (parts p).body (entry M dot p s).val
    apply Safety.rangeSafe_of_invariant (engine (withDot M dot)) bounds "start_n" p.length 64 (parts p).body
      (fun row t => ∃ a,Context M p s row a t)
    · rintro row hr t ⟨a,ctx⟩
      exact body_safe M dot p s t row a ctx hr bounds hb
    · rintro row hr t ⟨a,ctx⟩ u hu
      have eq := Option.some.inj ((bodyExec M dot p s t row a ctx).property.symm.trans hu)
      subst u
      exact ⟨update M dot p s row a,body_context M dot p s t row a ctx⟩
    · exact ⟨initialAcc M,entry_context M dot p s⟩
  · intro u hu
    obtain ⟨v,hv,row,ctx⟩ := loopExec M dot p s (entry M dot p s).val (entry_context M dot p s)
    simp only [step] at hu
    change range (engine (withDot M dot)) "start_n" 0 p.length 64 (parts p).body (entry M dot p s).val=some u at hu
    have eq := Option.some.inj (hv.symm.trans hu)
    subst u
    exact post_safe M dot p s v row _ ctx bounds hb

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Access
