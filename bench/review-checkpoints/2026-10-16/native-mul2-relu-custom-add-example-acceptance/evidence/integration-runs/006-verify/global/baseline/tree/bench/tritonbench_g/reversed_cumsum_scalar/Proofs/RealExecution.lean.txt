import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealInit

/-! Successful execution of the complete source and candidate at every time
length, including zero. The result is the independent padded suffix together
with a frame over all tagged memory cells outside the output row. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealExecution
open VeriTile.Triton RealProgram RealValues RealLoop
noncomputable section
set_option maxHeartbeats 2400000
set_option maxRecDepth 18000

theorem mirror_invariant (initial : BlockState) (T BT row : Nat) (s o : RegionName)
    (a : Nat → ℝ) (c : Nat) (u : BlockState) (h : Holds initial T BT row s o a c u)
    (j : Int) :
    Holds initial T BT row s o a c (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) := by
  refine ⟨h.undef, h.count_le, ?_, ?_, h.input, h.output, h.frame⟩
  · simpa using h.head
  · simpa using h.carry

theorem iteration (initial : BlockState) (swapped : Bool) (T BT row : Nat)
    (s o : RegionName) (a : Nat → ℝ) (hSO : s ≠ o) (j : Int) (u : BlockState)
    (hj : j < 1) (h : Holds initial T BT row s o a (SignedIndex.remaining j) u) :
    ∃ v, stepStmts (loopBody swapped T BT s o)
        (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) = some v ∧
      Holds initial T BT row s o a (SignedIndex.remaining (j + 1)) v := by
  have hm := mirror_invariant initial T BT row s o a (SignedIndex.remaining j) u h j
  rw [SignedIndex.current_chunk j hj] at hm
  obtain ⟨v, he, hv⟩ := RealLoop.iteration initial swapped T BT row s o a hSO j
    (u.setReg "__rev_i_t" .int [] (Tile.scalar j)) hj (by simp) hm
  exact ⟨v, he, by simpa only [SignedIndex.after_step] using hv⟩

theorem statement_eval (swapped : Bool) (T BT : Nat) (s o : RegionName) (u : BlockState) :
    stepStmt (statement swapped T BT s o) u =
      stepForRangeIntAux "__rev_i_t" (1 - (numChunks T BT : Int)) 1 1
        (loopBody swapped T BT s o) u := by
  rw [statement, stepForRangeIntAux.forRangeInt_unfold, start_eval]
  simp only [constInt_eval, evalOp_constNat, Option.bind_some, Option.bind_eq_bind, Tile.scalar_data]

theorem loop_executes (initial : BlockState) (swapped : Bool) (T BT row : Nat)
    (s o : RegionName) (a : Nat → ℝ) (hSO : s ≠ o) (u : BlockState)
    (h : Holds initial T BT row s o a (numChunks T BT) u) :
    ∃ v, stepStmt (statement swapped T BT s o) u = some v ∧
      Holds initial T BT row s o a 0 v := by
  obtain ⟨final, v, he, hf, hv⟩ := ReversedCumsum.SignedRange.executes "__rev_i_t" 1 1
    (loopBody swapped T BT s o)
    (fun j v => Holds initial T BT row s o a (SignedIndex.remaining j) v)
    (by decide) (fun j v hj hv => iteration initial swapped T BT row s o a hSO j v hj hv)
    (1 - (numChunks T BT : Int)) u (by simpa only [SignedIndex.at_start] using h)
  refine ⟨v, (statement_eval swapped T BT s o u).trans he, ?_⟩
  simpa only [SignedIndex.at_or_after_end final hf] using hv

def Post (T BT : Nat) (s o : RegionName) (initial final : BlockState) : Prop :=
  (∀ t, t < T → final.mem o (address T (initial.pids 0) t) =
    MemCell.real (expected T BT (initial.pids 0) (initial.readMem s) (initial.undef s) t)) ∧
  BlockState.WriteWithin (RealMemory.footprint T (initial.pids 0) o) initial final

theorem source_correct (swapped : Bool) (T BT : Nat) (s o : RegionName)
    (dtype out_dtype : FloatDType) (initial : BlockState) (hSO : s ≠ o) (hBT : 0 < BT) :
    ∃ final, exec (TypedSource.source swapped dtype out_dtype s o T BT).eraseDType.toAlgKernel initial =
      some final ∧ Post T BT s o initial final := by
  obtain ⟨final, he, h⟩ := loop_executes initial swapped T BT (initial.pids 0) s o
    (initial.readMem s) hSO (RealInit.initial initial)
    (RealInit.invariant T BT initial s o (initial.readMem s) (fun _ => rfl) hBT)
  refine ⟨final, ?_, ?_, h.frame⟩
  · rw [exec, source_body, body, ProgramRewrite.append, RealInit.executes]
    change stepStmts [statement swapped T BT s o] (RealInit.initial initial) = some final
    rw [stepStmts.cons_some he, stepStmts.nil]
  · intro t ht
    exact h.output t ht (by simp)

#axiomsClean iteration
#axiomsClean loop_executes
#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealExecution
