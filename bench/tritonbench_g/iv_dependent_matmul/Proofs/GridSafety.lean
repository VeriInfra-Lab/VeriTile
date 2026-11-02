import bench.tritonbench_g.iv_dependent_matmul.Proofs.SourceSafety
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Interface

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.SourceSafety
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul.SourceArithmetic
open TritonBenchG.IvDependentMatmul FPValue Kernels Whole
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Backing extents for every active input and output cell. -/
structure Windows (r : Request) (bounds : RegionName → Nat) : Prop where
  input : ∀ o, o<r.rows*r.inner → o<bounds r.A
  weight : ∀ o, o<r.inner*r.columns → o<bounds r.B
  output : ∀ o, o<r.rows*r.columns → o<bounds r.C

/-- Payload-free witness used only to reuse the already proved natural-index
bounds. No execution state or tensor condition is assumed by this witness. -/
def geometryState : State Word :=
  ⟨fun _ _ => Cell.mk .fp32 (FP.Value.zero .fp32),fun _ _ _ => none,
    fun _ => 0,fun _ => 0,fun d _ _ => defaultValue d⟩

private theorem append_safe (M : Algebra α) (bounds : RegionName → Nat)
    (a b : List ComputeStmt) (s t : State α)
    (ha : Safety.traceSafeList M bounds a s) (ran : run M a s=some t)
    (hb : Safety.traceSafeList M bounds b t) : Safety.traceSafeList M bounds (a++b) s := by
  rw [Safety.traceSafeList_append]
  refine ⟨ha,?_⟩
  intro u hu
  have eq := Option.some.inj (ran.symm.trans hu)
  simpa only [←eq] using hb

private theorem range_safe (M : Algebra α) (bounds : RegionName → Nat)
    (idx : RegName) (body : List ComputeStmt) (stop : Nat) (P : Nat → State α → Prop)
    (one : ∀ i, i<stop → ∀ s, P i s → ∃ q,
      run M body (s.setReg idx .int [] (fun _ => (i : Int)))=some q ∧ P (i+1) q)
    (safe : ∀ i, i<stop → ∀ s, P i s →
      Safety.traceSafeList M bounds body (s.setReg idx .int [] (fun _ => (i : Int))))
    (s : State α) (start : Nat) (bound : start≤stop) (initial : P start s) :
    Safety.rangeIntSafe M bounds idx (start : Int) (stop : Int) 1 body s := by
  have go : ∀ n i, i+n=stop → ∀ s, P i s → Safety.rangeIntSafe M bounds idx (i : Int) (stop : Int) 1 body s := by
    intro n
    induction n with
    | zero =>
      intro i hi s hs
      have eq : i=stop := by omega
      subst i
      rw [Safety.rangeIntSafe_unfold]
      simp
    | succ n ih =>
      intro i hi s hs
      have lt : i<stop := by omega
      obtain ⟨q,hq,hP⟩ := one i lt s hs
      rw [Safety.rangeIntSafe_unfold]
      simpa only [show ¬(1:Nat)=0 by omega,if_false,show (i : Int) < (stop : Int) by exact_mod_cast lt,if_true,hq,← Int.natCast_add] using
        And.intro (safe i lt s hs) (ih (i+1) (by omega) q hP)
  exact go (stop-start) start (by omega) s initial

/-- Access safety of the actual real quotation, including all five pointer
schedules, the complete dynamic K loop and the final masked cast/store. -/
theorem real_safe (rest : Algebra ℝ) (r : Request) (graph : Graph)
    (bounds : RegionName → Nat) (W : Windows r bounds) (pid : Nat)
    (hp : pid<programs r) (initial : State ℝ) :
    TraceSafe (RealExecution.backend rest) bounds (Kernel.quotation r.schedule graph r.parameters)
      {initial with pids:=fun ax => if ax=0 then pid else 0} := by
  let s : State ℝ := {initial with pids:=fun ax => if ax=0 then pid else 0}
  let p := r.parameters
  obtain ⟨t,ht,htr⟩ := RealExecution.prologueWithPointers_run rest r.schedule p s
  let P := fun cur t => ∃ acc, RealExecution.ReadyAt r.schedule p s t cur acc
  have one : ∀ cur, cur<RealExecution.count p → ∀ u, P cur u → ∃ v,
      run (RealExecution.backend rest) (Kernels.body r.schedule graph p)
        (u.setReg "k" .int [] (fun _ => (cur : Int)))=some v ∧ P (cur+1) v := by
    intro cur _ u ⟨acc,h⟩
    obtain ⟨v,hv,hR⟩ := RealExecution.schedule_body_run rest r.schedule graph p s u cur acc h
    exact ⟨v,hv,_,hR⟩
  have safe : ∀ cur, cur<RealExecution.count p → ∀ u, P cur u →
      Safety.traceSafeList (RealExecution.backend rest) bounds (Kernels.body r.schedule graph p)
        (u.setReg "k" .int [] (fun _ => (cur : Int))) := by
    intro cur _ u ⟨acc,h⟩
    apply body_safe _ _ r.schedule graph p u cur (RealExecution.aBase p s) (RealExecution.bBase p s)
      h.1.aptr h.1.bptr h.1.offk
    · simpa only [Pointers,RealExecution.Pointers,RealExecution.aPointerAt,RealExecution.bPointerAt,
        RealExecution.aOffset,RealExecution.bOffset] using h.2
    · intro i hi
      apply W.input
      simpa only [FPValue.aOffset,FPValue.aBase,FPValue.row,FPValue.rowBlock,FPValue.columnsGrid,
        RealExecution.aBase,RealExecution.row,RealExecution.rowBlock,RealExecution.columnsGrid,
        instanceState,s,p,Request.parameters] using loadA_bound r geometryState pid cur hp i hi
    · intro i hi
      apply W.weight
      simpa only [FPValue.bOffset,FPValue.bBase,FPValue.col,FPValue.colBlock,FPValue.columnsGrid,
        RealExecution.bBase,RealExecution.col,RealExecution.colBlock,RealExecution.columnsGrid,
        instanceState,s,p,Request.parameters] using loadB_bound r geometryState pid cur hp i hi
  have hs := range_safe (RealExecution.backend rest) bounds "k" (Kernels.body r.schedule graph p)
    (RealExecution.count p) P one safe t 0 (by omega) ⟨_,htr⟩
  obtain ⟨u,hu,hur⟩ := RealExecution.schedule_loop_run rest r.schedule graph p s t 0 _ htr
  have ran : run (RealExecution.backend rest)
      [.forRangeInt "k" (.constInt 0) (stopOp p) (.constNat 1) (Kernels.body r.schedule graph p)] t=some u := by
    rw [SourceArithmetic.loop_run (RealExecution.backend rest) p (Kernels.body r.schedule graph p) t]
    exact hu
  have loopSafe : Safety.traceSafeList (RealExecution.backend rest) bounds
      [.forRangeInt "k" (.constInt 0) (stopOp p) (.constNat 1) (Kernels.body r.schedule graph p)] t := by
    apply Safety.traceSafeList_cons_intro
    · simpa [Safety.stepSafe_unfold,stopOp_safe (RealExecution.backend rest) bounds p t,Safety.safeOp_unfold,SourceArithmetic.stopOp_eval (RealExecution.backend rest) p t,evalOp_unfold,numeric,bop,
        Option.bind,RealExecution.count] using hs
    · intro v _; exact Safety.traceSafeList_nil _ _ _
  rw [Kernel.quotation_eq]
  change Safety.traceSafeList _ bounds (Kernels.source r.schedule graph p).surfaceBody s
  rw [Kernels.source_split]
  apply append_safe _ _ _ _ _ _
    (append_safe _ _ _ _ _ _ (prologue_safe _ _ _ _ _) ht loopSafe)
    (by rw [run_append,ht];exact ran)
  apply suffix_safe _ _ p u _ _ _ hur.pm hur.pn hur.acc
  intro i j hi hj
  apply W.output
  have h := output_bound r _ _ hi hj
  simpa only [p,Request.parameters,RealExecution.rowBlock,RealExecution.colBlock,RealExecution.columnsGrid,s,
    offset,Nat.one_mul,Nat.mul_comm] using h

#axiomsClean real_safe

/-- Access safety of the actual native quotation, including all five pointer
schedules, the complete dynamic K loop and the final masked cast/store. -/
theorem native_safe (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc) (r : Request) (graph : Graph)
    (bounds : RegionName → Nat) (W : Windows r bounds) (pid : Nat)
    (hp : pid<programs r) (initial : State Word) :
    TraceSafe (FPValue.backend rest cfg D F) bounds (Kernel.quotation r.schedule graph r.parameters)
      {initial with pids:=fun ax => if ax=0 then pid else 0} := by
  let s : State Word := {initial with pids:=fun ax => if ax=0 then pid else 0}
  let p := r.parameters
  obtain ⟨t,ht,htr⟩ := Schedules.prologueWithPointers_run rest cfg D F r.schedule p s
  let P := fun cur t => ∃ acc, Schedules.ReadyAt r.schedule p s t cur acc
  have one : ∀ cur, cur<FPValue.count p → ∀ u, P cur u → ∃ v,
      run (FPValue.backend rest cfg D F) (Kernels.body r.schedule graph p)
        (u.setReg "k" .int [] (fun _ => (cur : Int)))=some v ∧ P (cur+1) v := by
    intro cur _ u ⟨acc,h⟩
    obtain ⟨v,hv,hR⟩ := Schedules.body_run rest cfg D F r.schedule graph p s u cur acc h
    exact ⟨v,hv,_,hR⟩
  have safe : ∀ cur, cur<FPValue.count p → ∀ u, P cur u →
      Safety.traceSafeList (FPValue.backend rest cfg D F) bounds (Kernels.body r.schedule graph p)
        (u.setReg "k" .int [] (fun _ => (cur : Int))) := by
    intro cur _ u ⟨acc,h⟩
    apply body_safe _ _ r.schedule graph p u cur (FPValue.aBase p s) (FPValue.bBase p s)
      h.1.aptr h.1.bptr h.1.offk
    · simpa only [Pointers,Schedules.Pointers,Schedules.aPointerAt,Schedules.bPointerAt,
        FPValue.aOffset,FPValue.bOffset] using h.2
    · intro i hi
      apply W.input
      simpa only [FPValue.aOffset,FPValue.aBase,FPValue.row,FPValue.rowBlock,FPValue.columnsGrid,
        FPValue.aBase,FPValue.row,FPValue.rowBlock,FPValue.columnsGrid,
        instanceState,s,p,Request.parameters] using loadA_bound r geometryState pid cur hp i hi
    · intro i hi
      apply W.weight
      simpa only [FPValue.bOffset,FPValue.bBase,FPValue.col,FPValue.colBlock,FPValue.columnsGrid,
        FPValue.bBase,FPValue.col,FPValue.colBlock,FPValue.columnsGrid,
        instanceState,s,p,Request.parameters] using loadB_bound r geometryState pid cur hp i hi
  have hs := range_safe (FPValue.backend rest cfg D F) bounds "k" (Kernels.body r.schedule graph p)
    (FPValue.count p) P one safe t 0 (by omega) ⟨_,htr⟩
  obtain ⟨u,hu,hur⟩ := Schedules.loop_run rest cfg D F r.schedule graph p s t 0 _ htr
  have ran : run (FPValue.backend rest cfg D F)
      [.forRangeInt "k" (.constInt 0) (stopOp p) (.constNat 1) (Kernels.body r.schedule graph p)] t=some u := by
    rw [SourceArithmetic.loop_run (FPValue.backend rest cfg D F) p (Kernels.body r.schedule graph p) t]
    exact hu
  have loopSafe : Safety.traceSafeList (FPValue.backend rest cfg D F) bounds
      [.forRangeInt "k" (.constInt 0) (stopOp p) (.constNat 1) (Kernels.body r.schedule graph p)] t := by
    apply Safety.traceSafeList_cons_intro
    · simpa [Safety.stepSafe_unfold,stopOp_safe (FPValue.backend rest cfg D F) bounds p t,Safety.safeOp_unfold,SourceArithmetic.stopOp_eval (FPValue.backend rest cfg D F) p t,evalOp_unfold,numeric,bop,
        Option.bind,FPValue.count] using hs
    · intro v _; exact Safety.traceSafeList_nil _ _ _
  rw [Kernel.quotation_eq]
  change Safety.traceSafeList _ bounds (Kernels.source r.schedule graph p).surfaceBody s
  rw [Kernels.source_split]
  apply append_safe _ _ _ _ _ _
    (append_safe _ _ _ _ _ _ (prologue_safe _ _ _ _ _) ht loopSafe)
    (by rw [run_append,ht];exact ran)
  apply suffix_safe _ _ p u _ _ _ hur.pm hur.pn hur.acc
  intro i j hi hj
  apply W.output
  have h := output_bound r _ _ hi hj
  simpa only [p,Request.parameters,FPValue.rowBlock,FPValue.colBlock,FPValue.columnsGrid,s,
    offset,Nat.one_mul,Nat.mul_comm] using h

#axiomsClean native_safe
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.SourceSafety
