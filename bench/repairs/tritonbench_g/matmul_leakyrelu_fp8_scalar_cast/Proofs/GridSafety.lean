import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.SourceSafety
import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.IO

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.SourceSafety
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 Execution
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Windows (p : Problem) (b : Buffers) (bounds : RegionName → Nat) : Prop where
  inputA : ∀ i, i<p.M → ∀ k, k<p.K → i*b.A.rowStride+k*b.A.columnStride<bounds "A"
  inputB : ∀ k, k<p.K → ∀ j, j<p.N → k*b.B.rowStride+j*b.B.columnStride<bounds "B"
  output : ∀ i, i<p.M → ∀ j, j<p.N → i*b.C.rowStride+j*b.C.columnStride<bounds "C"

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
      run M body (s.setReg idx .nat [] (fun _ => i))=some q ∧ P (i+1) q)
    (safe : ∀ i, i<stop → ∀ s, P i s →
      Safety.traceSafeList M bounds body (s.setReg idx .nat [] (fun _ => i)))
    (s : State α) (start : Nat) (bound : start≤stop) (initial : P start s) :
    Safety.rangeSafe M bounds idx start stop 1 body s := by
  have go : ∀ n i, i+n=stop → ∀ s, P i s → Safety.rangeSafe M bounds idx i stop 1 body s := by
    intro n
    induction n with
    | zero =>
      intro i hi s hs
      have eq : i=stop := by omega
      subst i
      rw [Safety.rangeSafe_unfold]
      simp
    | succ n ih =>
      intro i hi s hs
      have lt : i<stop := by omega
      obtain ⟨q,hq,hP⟩ := one i lt s hs
      rw [Safety.rangeSafe_unfold]
      simpa only [show ¬(1:Nat)=0 by omega,if_false,lt,if_true,hq] using
        And.intro (safe i lt s hs) (ih (i+1) (by omega) q hP)
  exact go (stop-start) start (by omega) s initial


/-- Full dynamic K-loop safety for the actual grouped PID, following the
already proved Context/HasAcc successors. Both activation branches are kept. -/
theorem kernel_safe (M : Algebra α) (D : DotAcc α) (p : Problem) (b : Buffers)
    (bounds : RegionName → Nat) (W : Windows p b bounds) (s : State α)
    (ht : s.pids 0<p.tiles) : TraceSafe (backend M D) bounds (source p b) s := by
  have hm : 0<p.M := by
    by_contra h
    have eq : p.M=0 := by omega
    simp [Problem.tiles,Problem.rows,eq,
      VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealMath.blocks] at ht
  have hn : 0<p.N := by
    by_contra h
    have eq : p.N=0 := by omega
    simp [Problem.tiles,Problem.columns,eq,
      VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealMath.blocks] at ht
  let start := preludeExecution M D p b s
  have ctx := preludeContext M D p b s
  let P := fun (_ : Nat) t => Context p s t ∧ ∃ acc, HasAcc acc t
  have one : ∀ k, k<trips p → ∀ t, P k t → ∃ u,
      run (backend M D) (body p b) (t.setReg "k" .nat [] (fun _ => k))=some u ∧ P (k+1) u := by
    intro k _ t ⟨hc,acc,ha⟩
    let next := bodyExecution M D p b s t k acc hc ha
    have h := bodyContext M D p b s t k acc hc ha
    exact ⟨next.val,next.property,h.1,_,h.2⟩
  have safe : ∀ k, k<trips p → ∀ t, P k t →
      Safety.traceSafeList (backend M D) bounds (body p b)
        (t.setReg "k" .nat [] (fun _ => k)) := by
    intro k _ t ⟨hc,acc,ha⟩
    apply body_safe M D p b s t k acc hc ha bounds
    · intro i hi
      rw [aOffset_eq p b s k i ht]
      have hk : k*64+i.2.1.val<p.K := by omega
      simp only [clamp,if_pos hk]
      exact W.inputA _ (Nat.mod_lt _ hm) _ hk
    · intro i hi
      rw [bOffset_eq p b s k i ht]
      have hk : k*64+i.1.val<p.K := by omega
      simp only [clamp,if_pos hk]
      exact W.inputB _ hk _ (Nat.mod_lt _ hn)
  have hs := range_safe (backend M D) bounds "k" (body p b) (trips p) P one safe
    start.val 0 (by omega) ⟨ctx.1,_,ctx.2⟩
  obtain ⟨u,hu,hc,ha⟩ := loopExecution M D p b s start.val ctx.1 ctx.2
  have kt : start.val.regs .nat [] "k_tiles"=some (fun _ => trips p) := ctx.1.kt
  have ran : run (backend M D)
      [.forRangeDyn "k" (.constNat 0) (.ref .nat [] "k_tiles") (.constNat 1) (body p b)] start.val=some u := by
    simpa [run,step,evalOp_unfold,kt] using hu
  have loopSafe : Safety.traceSafeList (backend M D) bounds
      [.forRangeDyn "k" (.constNat 0) (.ref .nat [] "k_tiles") (.constNat 1) (body p b)] start.val := by
    apply Safety.traceSafeList_cons_intro
    · simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,kt] using hs
    · intro t _;exact Safety.traceSafeList_nil _ _ _
  unfold TraceSafe
  rw [source_decomposition]
  apply append_safe _ _ _ _ _ _
    (append_safe _ _ _ _ _ _ (prelude_safe M D p b bounds s) start.property loopSafe)
    (by rw [run_append,start.property];exact ran)
  apply epilogue_safe M D p b s u _ hc ha bounds
  intro i hi
  rw [cOffset_eq p b s i ht hi]
  have hh := (active_eq p s i ht).mp hi
  exact W.output _ hh.1 _ hh.2

#axiomsClean kernel_safe
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.SourceSafety
