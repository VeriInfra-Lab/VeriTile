import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardRun

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- Both reductions are sums, so the real execution is total even at D=0. -/
theorem source_exec_isSome (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) (s : BlockState) :
    ∃ s', exec (source GradIn GradOut Out bm bn gm gn om on K D log causal fp16).toAlgKernel s = some s' := by
  exact ⟨(source_runState GradIn GradOut Out bm bn gm gn om on K D log causal fp16 s).val,
    (source_runState GradIn GradOut Out bm bn gm gn om on K D log causal fp16 s).property⟩

theorem source_frame (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) (s s' : BlockState)
    (hExec : exec (source GradIn GradOut Out bm bn gm gn om on K D log causal fp16).toAlgKernel s = some s')
    (r : RegionName) (offset : Nat)
    (hmiss : ∀ i : Fin D, i.val < K → ¬(GradIn = r ∧ s.pids 0 * bm + s.pids 1 * bn + i.val = offset)) :
    s'.mem r offset = s.mem r offset := by
  have hstate := Option.some.inj
    ((source_runState GradIn GradOut Out bm bn gm gn om on K D log causal fp16 s).property.symm.trans hExec)
  subst s'
  cases log <;> cases causal <;> cases fp16
  all_goals
    dsimp only [source_runState]
    refine Eq.trans (foldl_store_preserve_cell _ _ _ r offset _ _ ?_) rfl
    intro j _ hm hc
    exact hmiss j.1 (by simpa using hm) hc

private theorem causal_zero_padding {α : Type} (lane K row : Nat) (x z : α) :
    (if row < lane then z else if lane < K ∧ lane ≤ row then x else z) =
      (if lane < K ∧ lane ≤ row then x else z) := by
  by_cases h : lane ≤ row
  · simp [h, Nat.not_lt.mpr h]
  · have hl : row < lane := Nat.lt_of_not_ge h
    simp [h, hl]

/-- Whole-source readback for all eight flag profiles, all 2D program IDs,
three independent stride pairs and the original distinct read/write masks. -/
theorem source_correct (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool) (s s' : BlockState)
    (hExec : exec (source GradIn GradOut Out bm bn gm gn om on K D log causal fp16).toAlgKernel s = some s') :
    ∀ i : Fin D,
      s'.readMem GradIn (s.pids 0 * bm + s.pids 1 * bn + i.val) =
        if i.val < K then
          formula K (s.pids 1) log causal
            (fun j => s.readMem GradOut (s.pids 0 * gm + s.pids 1 * gn + j.val))
            (fun j => s.readMem Out (s.pids 0 * om + s.pids 1 * on + j.val)) i
        else s.readMem GradIn (s.pids 0 * bm + s.pids 1 * bn + i.val) := by
  have hstate := Option.some.inj
    ((source_runState GradIn GradOut Out bm bn gm gn om on K D log causal fp16 s).property.symm.trans hExec)
  subst s'
  intro i
  cases log <;> cases causal <;> cases fp16
  all_goals
    dsimp only [source_runState]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
      (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < K
    · have hzero : (0.0 : ℝ) = 0 := by norm_num
      by_cases hrow : i.val ≤ s.pids 1
      · have hfuture := Nat.not_lt.mpr hrow
        simp (config := { contextual := Bool.true })
          [hi, hzero, hrow, hfuture, formula, masked, readActive, tile_elementwise,
            causal_zero_padding, MaskedReduction.ite_some_some,
            WithBot.sum_someTerm_eq_some]
        all_goals first | rfl | exact Or.inl rfl
      · have hfuture := Nat.lt_of_not_ge hrow
        simp (config := { contextual := Bool.true })
          [hi, hzero, hrow, hfuture, formula, masked, readActive, tile_elementwise,
            causal_zero_padding, MaskedReduction.ite_some_some,
            WithBot.sum_someTerm_eq_some]
        all_goals first | rfl | exact Or.inl rfl
    · simp [hi]

/-- Region-model successful IO and the full unwritten-memory frame. -/
theorem source_region_run (GradIn GradOut Out : RegionName)
    (bm bn gm gn om on K D : Nat) (log causal fp16 : Bool)
    (s : BlockState) (g o : Fin D → ℝ)
    (hg : ∀ j, readActive K (s.pids 1) causal j.val = Bool.true →
      s.readMem GradOut (s.pids 0 * gm + s.pids 1 * gn + j.val) = g j)
    (ho : ∀ j, readActive K (s.pids 1) causal j.val = Bool.true →
      s.readMem Out (s.pids 0 * om + s.pids 1 * on + j.val) = o j) :
    ∃ s', exec (source GradIn GradOut Out bm bn gm gn om on K D log causal fp16).toAlgKernel s = some s'
      ∧ (∀ i : Fin D, i.val < K →
        s'.readMem GradIn (s.pids 0 * bm + s.pids 1 * bn + i.val) =
          formula K (s.pids 1) log causal g o i)
      ∧ (∀ r offset,
        (r ≠ GradIn ∨ ∀ i : Fin D, i.val < K → offset ≠ s.pids 0 * bm + s.pids 1 * bn + i.val) →
        s'.mem r offset = s.mem r offset) := by
  obtain ⟨s', hrun⟩ := source_exec_isSome GradIn GradOut Out bm bn gm gn om on K D log causal fp16 s
  refine ⟨s', hrun, ?_, ?_⟩
  · intro i hi
    have hv := source_correct GradIn GradOut Out bm bn gm gn om on K D log causal fp16 s s' hrun i
    rw [if_pos hi] at hv
    exact hv.trans (formula_congr K (s.pids 1) log causal _ g _ o hg ho i)
  · intro r offset hmiss
    apply source_frame GradIn GradOut Out bm bn gm gn om on K D log causal fp16 s s' hrun r offset
    intro i hi ⟨hr, hoff⟩
    rcases hmiss with hregion | hlane
    · exact hregion hr.symm
    · exact hlane i hi hoff.symm

#axiomsClean source_exec_isSome
#axiomsClean source_frame
#axiomsClean source_correct
#axiomsClean source_region_run
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward
