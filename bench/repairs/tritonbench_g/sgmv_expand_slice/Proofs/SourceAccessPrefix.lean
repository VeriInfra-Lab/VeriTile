import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceExecution
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceAccess
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

theorem list_invariant {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (invariant : State α → Prop) (code : List ComputeStmt)
    (safe : ∀ st ∈ code, ∀ t, invariant t → Safety.stepSafe M bounds st t)
    (preserve : ∀ st ∈ code, ∀ t u, invariant t → FP.Structural.step M st t=some u → invariant u)
    (s : State α) (initial : invariant s) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s initial)
    intro u hu
    exact ih (fun q hq => safe q (by simp [hq]))
      (fun q hq => preserve q (by simp [hq])) u (preserve st (by simp) s u initial hu)

theorem assignment_register {α : Type} [Inhabited α] (M : Algebra α)
    (name : RegName) (e : ComputeExpr dtype shape) (s t : State α)
    (other : name≠"cur_batch") (value : Values α .int [])
    (input : s.regs .int [] "cur_batch"=some value)
    (execution : FP.Structural.step M (.assign dtype shape name e) s=some t) :
    t.regs .int [] "cur_batch"=some value := by
  simp only [FP.Structural.step] at execution
  cases he : evalExpr M e s with
  | none => simp [he] at execution
  | some v =>
    simp only [he,Option.bind_some,Option.some.injEq] at execution
    cases execution
    simpa [State.setReg,Ne.symm other] using input

theorem prelude_tail_safe {α : Type} [Inhabited α] (M : Algebra α)
    (p : KernelArgs) (s : State α) (bounds : RegionName → Nat)
    (bound : p.batch<bounds p.lengths)
    (initial : s.regs .int [] "cur_batch"=some (fun _=>(p.batch : Int))) :
    Safety.traceSafeList M bounds ((SourceParts.prelude primaryHalf p).drop 2) s := by
  apply list_invariant M bounds (fun t=>t.regs .int [] "cur_batch"=some (fun _=>(p.batch : Int))) _ _ _ s initial
  · intro st hs t ht
    simp only [SourceParts.prelude,SourceParts.outer,kernel,primaryHalf,sourceFace,
      sourceHalfSingle323232,ComputeKernel.surfaceBody,List.take,List.drop,
      List.mem_cons,List.not_mem_nil,or_false] at hs
    rcases hs with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
    all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold,
      Safety.safeCompute_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalOp_unfold,ht,Region.cast,bound]
  · intro st hs t u ht hu
    simp only [SourceParts.prelude,SourceParts.outer,kernel,primaryHalf,sourceFace,
      sourceHalfSingle323232,ComputeKernel.surfaceBody,List.take,List.drop,
      List.mem_cons,List.not_mem_nil,or_false] at hs
    rcases hs with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
    all_goals exact assignment_register M _ _ t u (by decide) _ ht hu

#axiomsClean prelude_tail_safe

theorem prelude_safe {α : Type} [Inhabited α] (M : Algebra α)
    (abi : SourceABI) (p : KernelArgs) (s : State α)
    (bounds : RegionName → Nat) (bound : p.batch < bounds p.lengths) :
    Safety.traceSafeList M bounds (SourceParts.prelude abi p) s := by
  rw [SourcePrelude.common,←List.take_append_drop 2 (SourceParts.prelude primaryHalf p),
    Safety.traceSafeList_append]
  constructor
  · simp [SourceParts.prelude,SourceParts.outer,kernel,primaryHalf,sourceFace,
      sourceHalfSingle323232,ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold,
      FP.Structural.step,evalExpr,evalOp_unfold,State.setReg]
  · intro t ht
    apply prelude_tail_safe M p t bounds bound
    simp [SourceParts.prelude,SourceParts.outer,kernel,primaryHalf,sourceFace,
      sourceHalfSingle323232,ComputeKernel.surfaceBody,FP.Structural.run,
      FP.Structural.step,evalExpr,evalOp_unfold,State.setReg] at ht
    cases ht
    simp [State.setReg]

#axiomsClean prelude_safe

theorem active_prefix_safe {α : Type} [Inhabited α] (M : Algebra α)
    (abi : SourceABI) (p : KernelArgs) (s : State α) (adapter : Nat)
    (bounds : RegionName → Nat) (bound : p.batch < bounds p.starts) :
    Safety.traceSafeList M bounds (SourceParts.activePrelude abi p)
      (SourceInit.guardedState p s adapter) := by
  rw [SourcePrelude.active_common]
  apply list_invariant M bounds (fun t=>t.regs .int [] "cur_batch"=some (fun _=>(p.batch : Int)))
  · intro st hs t ht
    simp only [SourceParts.activePrelude,SourceParts.active,SourceParts.afterTile,
      SourceParts.afterLength,SourceParts.thenBody,SourceParts.outer,kernel,primaryHalf,
      sourceFace,sourceHalfSingle323232,ComputeKernel.surfaceBody,List.take,
      List.getElem?_cons_zero,List.getElem?_cons_succ,List.mem_cons,List.not_mem_nil,or_false] at hs
    rcases hs with rfl|rfl|rfl|rfl|rfl|rfl|rfl
    all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold,
      Safety.safeCompute_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalOp_unfold,ht,Region.cast,bound]
  · intro st hs t u ht hu
    simp only [SourceParts.activePrelude,SourceParts.active,SourceParts.afterTile,
      SourceParts.afterLength,SourceParts.thenBody,SourceParts.outer,kernel,primaryHalf,
      sourceFace,sourceHalfSingle323232,ComputeKernel.surfaceBody,List.take,
      List.getElem?_cons_zero,List.getElem?_cons_succ,List.mem_cons,List.not_mem_nil,or_false] at hs
    rcases hs with rfl|rfl|rfl|rfl|rfl|rfl|rfl
    all_goals exact assignment_register M _ _ t u (by decide) _ ht hu
  · simp [SourceInit.guardedState,SourcePrelude.prepared,SourcePrelude.scalarBindings,State.setReg]

#axiomsClean active_prefix_safe

theorem guarded_safe {α : Type} [Inhabited α] (M : Algebra α)
    (abi : SourceABI) (p : KernelArgs) (code : List ComputeStmt) (s : State α)
    (length adapter : Nat) (tileRow : Nat)
    (hm : s.regs .int [] "M" = some (fun _ => (length : Int)))
    (hrow : s.regs .int [] "pid_m" = some (fun _ => (tileRow : Int)))
    (hbatch : s.regs .int [] "cur_batch" = some (fun _ => (p.batch : Int)))
    (hadapter : (s.mem p.adapters p.batch).read .int = (adapter : Int))
    (live : 0 < length) (tile : tileRow*32 < length)
    (bounds : RegionName → Nat) (bound : p.batch < bounds p.adapters)
    (safe : Safety.traceSafeList M bounds code
      (s.setReg "lora_index" .int [] (fun _ => (adapter : Int)))) :
    Safety.traceSafeList M bounds (SourceGuards.guarded abi p code) s := by
  rw [SourceGuards.common]
  have hl : ¬ (length : Int) ≤ 0 := by omega
  have ht : ¬ (length : Int) ≤ (tileRow : Int)*32 := by omega
  have ha : (adapter : Int) ≠ -1 := by omega
  simpa (maxSteps := 1000000) [SourceGuards.guarded,SourceParts.condition,
    SourceParts.afterTile,SourceParts.afterLength,SourceParts.thenBody,
    SourceParts.outer,kernel,primaryHalf,sourceFace,sourceHalfSingle323232,
    ComputeKernel.surfaceBody,ScalarFormat.dtype,LowFormat.scalar,
    Safety.traceSafeList_cons,Safety.traceSafeList_nil,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,FP.Structural.step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,
    numericLt,numericLe,numericEq,Region.cast,State.setReg,hm,hrow,hbatch,
    hadapter,hl,ht,ha,Nat.ne_of_gt live] using And.intro bound safe

#axiomsClean guarded_safe
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceAccess
