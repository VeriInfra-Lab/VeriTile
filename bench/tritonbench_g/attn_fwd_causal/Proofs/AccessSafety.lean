import bench.tritonbench_g.attn_fwd_causal.Proofs.Output
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.AttnFwdCausal.Source.AccessSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite Safety
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Bounds used by the actual source trace of one legal query tile. Masked
lanes do not require backing; the two scalar scale loads are unmasked. -/
structure Bounds (p : Parameters) (s : State α) (bounds : RegionName → Nat) : Prop where
  query : ∀ (i j : Fin 128), s.pids 0*128+i.val<p.length →
    base p s p.q+(s.pids 0*128+i.val)*p.q.row+j.val*p.q.dim<bounds p.Q
  queryScale : s.pids 1*((p.length+127)/128)+s.pids 0<bounds p.QS
  key : ∀ row, row<p.length → ∀ (j : Fin 128),
    base p s p.k+j.val*p.k.dim+row*p.k.row<bounds p.K
  value : ∀ row, row<p.length → ∀ (j : Fin 128),
    base p s p.v+row*p.v.row+j.val*p.v.dim<bounds p.V
  keyScale : ∀ row, row<p.length →
    s.pids 1*((p.length+63)/64)+row/64<bounds p.KS
  output : ∀ (i j : Fin 128), s.pids 0*128+i.val<p.length →
    base p s p.o+(s.pids 0*128+i.val)*p.o.row+j.val*p.o.dim<bounds p.Out

theorem before_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) (backed : Bounds p s bounds) :
    traceSafeList (engine (withDot M dot)) bounds (parts p).before s := by
  simp (config := {maxSteps:=300000}) [parts,kernel,TypedSource.repairedKernel,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,Broadcast.leftIndex,Broadcast.rightIndex]
  exact ⟨backed.query,backed.queryScale⟩

theorem body {phase : Bool} (M : Algebra α) (dot : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (backed : Bounds p initial bounds) (row : Nat) (a : Acc α) (hr : row<p.length)
    (ctx : Context M p initial row a current (phase:=phase)) :
    traceSafeList (engine (withDot M dot)) bounds (loopParts p phase).body
      (current.setReg (workName phase "start_n") .nat [] (fun _ => row)) := by
  have h_length:=ctx.length
  have h_startM:=ctx.startM
  have h_firstLo:=ctx.firstLo
  have h_firstHi:=ctx.firstHi
  have h_rows:=ctx.rows
  have h_cols:=ctx.cols
  have h_query:=ctx.query
  have h_scale:=ctx.scale
  have h_keyPointer:=ctx.keyPointer
  have h_valuePointer:=ctx.valuePointer
  have h_scalePointer:=ctx.scalePointer
  have h_maximum:=ctx.maximum
  have h_normalizer:=ctx.normalizer
  have h_value:=ctx.value
  cases phase <;> simp [workName] at h_length h_startM h_firstLo h_firstHi h_rows h_cols h_query h_scale h_keyPointer h_valuePointer h_scalePointer h_maximum h_normalizer h_value <;> simp (config := {maxSteps:=300000}) [loopParts,diagonal,parts,workName,kernel,TypedSource.repairedKernel,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,Broadcast.leftIndex,Broadcast.rightIndex,
    h_length,ctx.memory,h_startM,h_firstLo,h_firstHi,ctx.keyBase,ctx.valueBase,ctx.scaleBase,
    h_rows,h_cols,h_query,h_scale,h_keyPointer,h_valuePointer,
    h_scalePointer,ctx.outputPointer,h_maximum,h_normalizer,h_value]
  all_goals refine ⟨?_,backed.keyScale row hr,?_⟩
  all_goals
    intro i j active
    first
    | have hj : j.val<p.length-row := active
      exact backed.key (row+j.val) (by omega) i
    | have hi : i.val<p.length-row := active
      exact backed.value (row+i.val) (by omega) j

theorem middle (M : Algebra α) (dot : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    traceSafeList (engine (withDot M dot)) bounds (Source.middle p) current := by
  have h_maximum:=ctx.maximum
  have h_normalizer:=ctx.normalizer
  have h_value:=ctx.value
  simp [workName] at h_maximum h_normalizer h_value
  simp (config := {maxSteps:=300000}) [Source.middle,parts,kernel,TypedSource.repairedKernel,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    workName,ctx.callerStartM,ctx.callerRows,ctx.callerCols,ctx.callerQuery,ctx.callerScale,h_maximum,h_normalizer,h_value,ctx.startM,ctx.keyBase,ctx.valueBase,ctx.scaleBase]

theorem post (M : Algebra α) (dot : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (backed : Bounds p initial bounds) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current (phase:=Bool.true)) :
    traceSafeList (engine (withDot M dot)) bounds (diagonal p).after current := by
  have h_maximum:=ctx.maximum
  have h_normalizer:=ctx.normalizer
  have h_value:=ctx.value
  simp [workName] at h_maximum h_normalizer h_value
  simp (config := {maxSteps:=300000}) [diagonal,parts,kernel,TypedSource.repairedKernel,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,Broadcast.leftIndex,Broadcast.rightIndex,
    workName,h_maximum,ctx.callerRows,ctx.memory,ctx.rows,ctx.outputPointer,h_normalizer,h_value,FP.Structural.store]
  exact backed.output

theorem range_safe {phase : Bool} (M : Algebra α) (dot : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (backed : Bounds p initial bounds) (row finish : Nat) (a : Acc α)
    (hf : finish≤p.length) (ctx : Context M p initial row a current (phase:=phase)) :
    rangeSafe (engine (withDot M dot)) bounds (workName phase "start_n") row finish 64 (loopParts p phase).body current := by
  by_cases active : row<finish
  · rw [rangeSafe_unfold]
    simp only [show (64:Nat)≠0 by omega,active,if_false,if_true]
    refine ⟨body M dot p bounds initial current backed row a (by omega) ctx,?_⟩
    rw [(bodyExec M dot p initial current row a ctx).property]
    exact range_safe M dot p bounds initial _ backed (row+64) finish
      (update M dot p initial row a) hf (body_context M dot p initial current row a ctx)
  · rw [rangeSafe_unfold]
    simp [active]
termination_by finish-row
decreasing_by omega

theorem first_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (backed : Bounds p initial bounds) (a : Acc α)
    (legal : initial.pids 0*128<p.length) (ctx : Context M p initial 0 a current) :
    stepSafe (engine (withDot M dot)) bounds (firstStatement p) current := by
  have safe := range_safe M dot p bounds initial current backed 0 (initial.pids 0*128) a (by omega) ctx
  have lo:=ctx.firstLo
  have hi:=ctx.firstHi
  simp [workName] at lo hi
  simpa [workName,loopParts,firstStatement,stepSafe_unfold,safeOp_unfold,parts,kernel,TypedSource.repairedKernel,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    evalOp_unfold,lo,hi] using safe

theorem second_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (backed : Bounds p initial bounds) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    stepSafe (engine (withDot M dot)) bounds (secondStatement p)
      (middleExec M dot p initial current row a ctx).val := by
  have safe := range_safe M dot p bounds initial _ backed (initial.pids 0*128) (stop p initial) a
    (Nat.min_le_right _ _) (middle_context M dot p initial current row a ctx)
  have next:=middle_context M dot p initial current row a ctx
  have lo:=next.firstLo
  have hi:=next.firstHi
  simp [workName] at lo hi
  simpa [loopParts,workName,secondStatement,stepSafe_unfold,safeOp_unfold,
    diagonal,kernel,TypedSource.repairedKernel,ComputeKernel.surfaceBody,
    Triton.Source.dynamicRange?,Option.map,Option.getD,evalOp_unfold,
    lo,hi] using safe

/-- Safety of the exact repaired quotation, through both complete causal
ranges and the final output store, at every actual intermediate state. -/
theorem source_safe (M : Algebra α) (dot : Dot α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) (backed : Bounds p s bounds)
    (legal : s.pids 0*128<p.length) :
    TraceSafe (engine (withDot M dot)) bounds (kernel p) s := by
  let start:=entry M dot p s
  obtain ⟨mid,hm,row,hc⟩:=loopAt M dot p s start.val 0 (s.pids 0*128) (initialAcc M) (entry_context M dot p s)
  let reset:=middleExec M dot p s mid row (prefixAcc M dot p s) hc
  obtain ⟨last,hl,lastrow,hlc⟩:=loopAt M dot p s reset.val (s.pids 0*128) (stop p s)
    (prefixAcc M dot p s) (middle_context M dot p s mid row _ hc)
  have hfirst : step (engine (withDot M dot)) (firstStatement p) start.val=some mid :=
    (first_step M dot p s start.val 0 _ (entry_context M dot p s)).trans hm
  have hsecond : step (engine (withDot M dot)) (secondStatement p) reset.val=some last :=
    (second_step M dot p s mid row _ hc).trans hl
  unfold TraceSafe
  rw [source_body]
  simp only [List.append_assoc,List.singleton_append]
  rw [traceSafeList_append]
  refine ⟨before_safe M dot p bounds s backed,?_⟩
  intro next ran
  have same : next=start.val := Option.some.inj (ran.symm.trans start.property)
  subst next
  apply traceSafeList_cons_of_step (first_safe M dot p bounds s start.val backed _ legal (entry_context M dot p s)) hfirst
  apply (traceSafeList_append (engine (withDot M dot)) bounds (Source.middle p)
    (secondStatement p :: (diagonal p).after) mid).mpr
  refine ⟨middle M dot p bounds s mid row _ hc,?_⟩
  intro next ran
  have same : next=reset.val := Option.some.inj (ran.symm.trans reset.property)
  subst next
  exact traceSafeList_cons_of_step (second_safe M dot p bounds s mid backed row _ hc) hsecond
    (post M dot p bounds s last backed lastrow _ hlc)

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Source.AccessSafety
