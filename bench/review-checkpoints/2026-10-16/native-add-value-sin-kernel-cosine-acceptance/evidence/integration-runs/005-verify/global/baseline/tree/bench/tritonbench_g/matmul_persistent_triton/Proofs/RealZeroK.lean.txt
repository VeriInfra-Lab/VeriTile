import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealFull

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealZeroK
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

theorem count_zero (p : Parameters) (hK : p.K = 0) (hBK : 0 < p.BK) :
    RealLoopState.count p = 0 := by
  simp only [RealLoopState.count,RealMath.blocks,hK,Nat.zero_add]
  exact Nat.div_eq_of_lt (by omega)

theorem bound_zero (p : Parameters) (s : BlockState) (hK : p.K = 0) (hBK : 0 < p.BK) :
    evalOp RealLoopProgram.bound (RealPrefix.result p s) = some (Tile.scalar 0) := by
  simpa only [RealLoopState.stop,count_zero p hK hBK,Nat.zero_mul] using
    RealLoopProgram.bound_evaluates p s

theorem original_exec (p : Parameters) (s : BlockState) (hK : p.K = 0) (hBK : 0 < p.BK) :
    exec (RealSource.kernel p Bool.false) s = some (RealPrefix.result p s) := by
  have hloop : stepStmt (RealLoopProgram.loop p) (RealPrefix.result p s) =
      some (RealPrefix.result p s) := by
    rw [RealLoopProgram.loop,stepForRangeAux.forRangeDyn_unfold]
    simp only [evalOp_constNat,bound_zero p s hK hBK,Option.bind_some,Tile.scalar]
    exact stepForRangeAux.step_ge (by decide) (by omega)
  rw [exec,RealLoopProgram.matches_source,
    stepStmts.append_some (RealPrefix.executes p s)]
  simp only [stepStmts,hloop]

theorem original_safe (p : Parameters) (bounds : RegionBounds) (s : BlockState)
    (hK : p.K = 0) (hBK : 0 < p.BK) :
    (RealSource.kernel p Bool.false).TraceSafe bounds s := by
  rw [Kernel.TraceSafe,RealLoopProgram.matches_source]
  apply TraceProofs.append_safe _ _ (RealPureSafety.prefix_safe p bounds s)
  intro t ht
  rw [RealPrefix.executes] at ht
  cases ht
  apply Stmt.TraceSafeList.cons_intro
  · simp only [RealLoopProgram.loop,Stmt.TraceSafe,evalOp_constNat,bound_zero p s hK hBK,Tile.scalar]
    simp [RealLoopProgram.bound,Op.SafeAt.eq_def,Stmt.forRangeTraceSafe]
  · intro _ _
    exact Stmt.TraceSafeList.nil_intro

/-- The actual empty-reduction source never executes a store. In particular,
preexisting nonzero output cells remain nonzero; no zero-matrix postcondition
is substituted for this original source behavior. -/
theorem executes (p : Parameters) (v : Bool) (s : BlockState)
    (hK : p.K = 0) (hBK : 0 < p.BK) :
    exec (RealSource.kernel p v) s = some (RealPrefix.result p s) ∧
      (RealPrefix.result p s).mem = s.mem := by
  have hf := congrArg BlockState.mem (RealPrefix.frame p s)
  refine ⟨?_,hf⟩
  rw [RealVariant.equivalent]
  exact original_exec p s hK hBK

/-- The same complete memory preservation holds for finite flat allocations,
for both storage projections and source variants, from arbitrary memory. -/
theorem flat_unchanged (p : Parameters) (storage : TypedSource.Storage) (v : Bool)
    (A : FlatAlloc) (s : BlockState) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0) (hu : s.undef = (fun _ _ => 0))
    (hK : p.K = 0) (hBK : 0 < p.BK) :
    ∃ t, exec (A.flattenKernel (p.source storage v).eraseDType.toAlgKernel)
        (A.flattenState s) = some t ∧ t.mem = (A.flattenState s).mem := by
  rw [RealFull.algorithm]
  have hs := (RealVariant.safe_iff p v A.extent s).mpr (original_safe p A.extent s hK hBK)
  refine ⟨A.flattenState (RealPrefix.result p s),?_,?_⟩
  · rw [A.exec_flatten hd hc _ s hs (RealPureSafety.flatten_ok p v) hu,
      (executes p v s hK hBK).1,Option.map_some]
  · funext r o
    apply A.flattenState_frame s (RealPrefix.result p s) (fun _ _ => False)
    · intro r o _
      exact congrFun (congrFun (executes p v s hK hBK).2 r) o
    · exact Or.inr (by intro r o h; cases h)

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealZeroK
