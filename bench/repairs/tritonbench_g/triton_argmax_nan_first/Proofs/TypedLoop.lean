import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedStep

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Prefix values are derived from actual program IDs, including segmentation. -/
theorem prefix_run (rest : Algebra Word) (s : State Word) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) :
    ∃ t, run (backend rest) ((TypedSource.dimension X O M N K BM BN base index64).surfaceBody.take 7) s=some t ∧
      t.mem=s.mem ∧ Ready BM (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K)
        (fun _ => (0xff800000,0)) t := by
  have nonnegative : ¬ ((s.pids 0 + base : Nat) : Int) < 0 := by omega
  have remainder := Nat.mod_add_div (s.pids 0 + base) K
  cases index64 <;>
    simp [TypedSource.dimension,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      backend,State.setReg,Ready,ofFloat,toFloat,bop,numeric,numericLt,integralFloorDiv,integralMod,ComputeDType.eraseDType,Option.bind, IntegerExpr.sameShape, -Nat.cast_add, -Nat.cast_mul, ← Int.natCast_add, ← Int.natCast_mul, -Int.natCast_mod, ← Int.natCast_ediv, ← Int.natCast_emod]
  all_goals constructor <;> funext i <;> simp [bop, Broadcast.leftIndex, Broadcast.rightIndex, -Nat.cast_add, -Nat.cast_mul, ← Int.natCast_add, ← Int.natCast_mul, -Int.natCast_mod, ← Int.natCast_ediv, ← Int.natCast_emod, Int.emod_def, nonnegative]
  all_goals
    have equation := congrArg (fun n : Nat => (n : Int)) remainder
    simp only [Nat.cast_add] at equation
    exact sub_eq_iff_eq_add.mpr equation.symm

def vectorLoop (initial : State Word) (X : RegionName) (M N K BM BN row inner : Nat)
    (positive : 0<BN) (cur : Nat) (current : Fin BM → Word × Nat) : Fin BM → Word × Nat :=
  if h : cur<N then vectorLoop initial X M N K BM BN row inner positive (cur+BN)
    (next initial X M N K BM BN row inner cur positive current) else current
termination_by N-cur
decreasing_by omega

theorem next_memory (initial s : State Word) (same : s.mem=initial.mem) (X : RegionName)
    (M N K BM BN row inner cur : Nat) (positive : 0<BN) (current : Fin BM → Word × Nat) :
    next s X M N K BM BN row inner cur positive current=
      next initial X M N K BM BN row inner cur positive current := by
  funext i
  have loads : loaded s X M N K BM BN row inner cur i=loaded initial X M N K BM BN row inner cur i := by
    funext j
    simp only [loaded,same]
  simp only [next,loads,same]

theorem range_run (rest : Algebra Word) (initial s : State Word) (same : s.mem=initial.mem)
    (X : RegionName) (O : Region .int) (M N K BM BN base row inner cur : Nat)
    (index64 : Bool) (positive : 0<BN) (current : Fin BM → Word × Nat)
    (ready : Ready BM row inner current s) :
    ∃ t, range (backend rest) "start_n" cur N BN (body X O M N K BM BN base index64) s=some t ∧
      t.mem=initial.mem ∧ Ready BM row inner
        (vectorLoop initial X M N K BM BN row inner positive cur current) t := by
  by_cases active : cur<N
  · obtain ⟨middle,runStep,memStep,readyStep⟩ :=
      body_run rest s X O M N K BM BN base row inner cur index64 positive current ready
    rw [next_memory initial s same] at readyStep
    obtain ⟨t,runRest,memRest,readyRest⟩ := range_run rest initial middle (memStep.trans same)
      X O M N K BM BN base row inner (cur+BN) index64 positive _ readyStep
    refine ⟨t,?_,memRest,?_⟩
    · rw [range_step _ _ _ _ _ _ _ (Nat.ne_of_gt positive) active,runStep]
      exact runRest
    · rw [vectorLoop,dif_pos active]
      exact readyRest
  · refine ⟨s,range_done _ _ _ _ _ _ _ (by omega),same,?_⟩
    rw [vectorLoop,dif_neg active]
    exact ready
termination_by N-cur
decreasing_by omega

/- All loop iterations use the initial input memory because the quoted body
has no stores. Register shadowing is retained by the actual evaluator. -/
#axiomsClean prefix_run
#axiomsClean range_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Typed
