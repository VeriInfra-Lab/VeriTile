import bench.tritonbench_g.softmax_optimize.Proofs.FirstLoops
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SignedPrevious

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Reduction
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
set_option maxHeartbeats 4000000
noncomputable section

theorem previous_eval (N B : Nat) (hN : 0 < N) (hB : 0 < B) (s : BlockState)
    (hcols : s.regs .int [] "cols" = some (Tile.scalar (N : Int))) :
    evalOp (Source.previousExpression B) s = some (Tile.scalar (Config.previous N B : Int)) := by
  have he : evalOp (Source.previousExpression B) s =
      evalOp (FullSource.previousExpression N B) s := by
    simp only [Source.previousExpression, FullSource.previousExpression, FullSource.signedQuotient,
      evalOp.eq_def, hcols, Option.bind_some, Option.bind_eq_bind]
  rw [he, SignedTransport.previous_eval N B hN hB]
  rfl

def seedStatements (N B : Nat) : List Stmt :=
  [.assign .int [] "pid_m" (.castNatToInt (.programId 0)),
   .assign .int [] "cols" (.constInt N),
   .assign .real [B] "m" (.full [B] .negInf),
   .assign .real [B] "z" (.full [B] (.const 0)),
   .assign .int [] "prev_multiple" (Source.previousExpression B)]

def seed (s : BlockState) (N B : Nat) : BlockState :=
  ((((s.setReg "pid_m" .int [] (Tile.scalar (s.pid : Int))).setReg
    "cols" .int [] (Tile.scalar (N : Int))).setReg
    "m" .real [B] ⟨fun _ => (⊥ : WithBot ℝ)⟩).setReg
    "z" .real [B] ⟨fun _ => ((0 : ℝ) : WithBot ℝ)⟩).setReg
    "prev_multiple" .int [] (Tile.scalar (Config.previous N B : Int))

theorem seed_run (s : BlockState) (N B : Nat) (hN : 0 < N) (hB : 0 < B) :
    stepStmts (seedStatements N B) s = some (seed s N B) := by
  unfold seedStatements
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := Tile.scalar (s.pid : Int)) (by simp only [evalOp.eq_def, Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := Tile.scalar (N : Int))
    (by simp only [evalOp.eq_def]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := (⟨fun _ => (⊥ : WithBot ℝ)⟩ : Tile .real [B]))
    (by simp only [evalOp.eq_def, Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (v := (⟨fun _ => ((0 : ℝ) : WithBot ℝ)⟩ : Tile .real [B]))
    (by simp only [evalOp.eq_def, Option.bind_some]; rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (previous_eval N B hN hB _
    (by simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true])))]
  rw [stepStmts.nil]
  rfl

theorem seed_invariant (s : BlockState) (X : RegionName) (N B : Nat) :
    FirstPass.Invariant X N B s 0 (seed s N B) := by
  refine ⟨⟨rfl, rfl, rfl, rfl, ?_, ?_, ?_⟩, ?_, ?_⟩
  all_goals simp only [seed, FirstPass.maxima, FirstPass.masses, Recurrence.laneAfter,
    List.range_zero, List.foldl_nil, Recurrence.realOperations,
    BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq, not_false_eq_true]

/-- The exact seed and both actual signed loops establish the old mathematical
lane invariant constructively, ready for the unchanged source reduction. -/
theorem first_pass (s : BlockState) (X : RegionName) (N B : Nat) (hN : 0 < N) (hB : 0 < B) :
    ∃ t, stepStmts (seedStatements N B ++
      [FirstLoops.fullStatement X N B, FirstLoops.tailStatement X N B]) s = some t ∧
      FirstPass.Invariant X N B s (Config.fullTiles N B + 1) t := by
  rw [ProgramRewrite.append, seed_run s N B hN hB]
  simp only [Option.bind_some]
  exact FirstLoops.inputs_run X N B hN hB s _ (seed_invariant s X N B)

theorem reduction_run (initial current : BlockState) (X : RegionName) (N B : Nat)
    (hN : 0 < N) (hB : 0 < B)
    (h : FirstPass.Invariant X N B initial (Config.fullTiles N B + 1) current) :
    ∃ (shift : ℝ) (t : BlockState), stepStmts (reductionStmts B) current = some t ∧
      t.pids = current.pids ∧ t.mem = current.mem ∧
      (∀ rg o, t.undef rg o = current.undef rg o) ∧
      t.regs .real [] "m" = some (Tile.scalar (shift : WithBot ℝ)) ∧
      t.regs .real [] "z" = some (Tile.scalar
        ((Real.exp (-shift) * ∑ j ∈ Finset.range N,
          Real.exp (current.readMem X (current.pid * N + j)) : ℝ) : WithBot ℝ)) := by
  have hp : current.pid = initial.pid := by unfold BlockState.pid; rw [h.pids]
  have hc : ∀ (lane : Fin B) (count : Nat),
      laneState initial X N B lane count = laneState current X N B lane count := by
    intro lane count
    exact (laneState_congr current initial X N B hp h.mem lane count).symm
  have hm : current.regs .real [B] "m" = some
      ⟨fun i => (laneState initial X N B i.1 (Config.fullTiles N B + 1)).1⟩ := by
    rw [h.m]
    congr 1
    apply Tile.ext
    intro i
    exact congrArg Prod.fst (Recurrence.laneAfter_real initial X N B _ i.1)
  have hz : current.regs .real [B] "z" = some
      ⟨fun i => ((laneState initial X N B i.1 (Config.fullTiles N B + 1)).2 : WithBot ℝ)⟩ := by
    rw [h.z]
    congr 1
    apply Tile.ext
    intro i
    exact congrArg Prod.snd (Recurrence.laneAfter_real initial X N B _ i.1)
  obtain ⟨shift, t, ht, hpid, hmem, hundef, _, hm, hz⟩ :=
    reduction_step N B hN hB X (Config.fullTiles N B + 1) (by omega)
      (Config.tail_covers hN hB) initial current hp hc hm hz
  exact ⟨shift, t, ht, hpid, hmem, hundef, hm, hz⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.Reduction
