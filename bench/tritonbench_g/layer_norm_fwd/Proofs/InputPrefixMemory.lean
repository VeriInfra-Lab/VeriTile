import bench.tritonbench_g.layer_norm_fwd.Proofs.InputPrefix

/-! Exact tagged memory after the complete input prefix, including both
dropout masks and ResidualOut, with last-store order retained even when output
regions alias. All stored values use the initial mathematical input snapshot. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.InputPrefix
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem secondState_mem (f : Flags) (r : Regions) (stride : Strides)
    (M N block : Nat) (p : ℝ) (u : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) (region : RegionName) (offset : Nat) :
    (secondState f r stride M N block p u).mem region offset =
      if f.dropout && f.dropoutMask && f.x1 &&
          decide (region = r.dropoutMask ∧ Contract.inRow ((M+u.pid)*N) N offset) then
        MemCell.of .bool (Formula.keep (u.readMemValue .int r.seeds (M+u.pid)) p
          (offset-(M+u.pid)*N))
      else (Execution.firstStreamState f r stride N block p u).mem region offset := by
  cases hx : f.x1
  · simp [secondState, hx]
  · simp only [secondState, hx, ↓reduceIte]
    rw [InputSuffix.bodyState_mem _ _ _ _ _ _ _ _ _ _ hN]
    cases hd : f.dropout <;> cases hm : f.dropoutMask
    all_goals simp only [hd, hm, Bool.false_and, Bool.true_and, Bool.and_true,
      Bool.false_eq_true, if_false]
    rw [Domain.readsUnchanged_seed f r _ u
      (Execution.firstStreamState_readsUnchanged f r stride N block p u hN hsep) hd]

theorem state_readsUnchanged (f : Flags) (r : Regions) (stride : Strides)
    (M N block : Nat) (p : ℝ) (u : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) :
    Domain.ReadsUnchanged f r (state f r stride M N block p u) u := by
  have hprevious := afterResidual_readsUnchanged f r stride M N block p u hN hsep
  cases hf : f.residualOut
  · simpa [state, InputSuffix.residualOutState, hf] using hprevious
  · simp only [state, InputSuffix.residualOutState, hf, ↓reduceIte]
    apply Domain.readsUnchanged_trans f r _ _ _ _ hprevious
    apply Domain.scatter_readsUnchanged f r hsep r.residualOut
      (by simp [Domain.writeRegion, hf]) _ _ _ hN

theorem state_mem (f : Flags) (r : Regions) (stride : Strides)
    (M N block : Nat) (p : ℝ) (u : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) (region : RegionName) (offset : Nat) :
    (state f r stride M N block p u).mem region offset =
      if f.residualOut && decide (region = r.residualOut ∧
          Contract.inRow (u.pid * stride.residualOut) N offset) then
        MemCell.real (Formula.combined f (snapshot r stride u) M p u.pid
          (offset-u.pid*stride.residualOut))
      else if f.dropout && f.dropoutMask && f.x1 &&
          decide (region = r.dropoutMask ∧ Contract.inRow ((M+u.pid)*N) N offset) then
        MemCell.of .bool (Formula.keep (u.readMemValue .int r.seeds (M+u.pid)) p
          (offset-(M+u.pid)*N))
      else if f.dropout && f.dropoutMask &&
          decide (region = r.dropoutMask ∧ Contract.inRow (u.pid*N) N offset) then
        MemCell.of .bool (Formula.keep (u.readMemValue .int r.seeds u.pid) p
          (offset-u.pid*N))
      else u.mem region offset := by
  rw [state, InputSuffix.residualOutState_mem _ _ _ _ _ _ _ hN]
  by_cases hstore : (f.residualOut && decide (region = r.residualOut ∧
    Contract.inRow (u.pid * stride.residualOut) N offset)) = Bool.true
  · simp only [hstore, ↓reduceIte]
    have hparts : f.residualOut = Bool.true ∧
        (region = r.residualOut ∧ Contract.inRow (u.pid * stride.residualOut) N offset) := by
      simpa only [Bool.and_eq_true, decide_eq_true_eq] using hstore
    have hin := hparts.2.2
    have hcol : offset - u.pid * stride.residualOut < N := by
      unfold Contract.inRow at hin
      omega
    simp only [values, Formula.combined_padded, hcol, if_true]
  · simp only [hstore, Bool.false_eq_true, if_false, afterResidual, InputSuffix.residualState_mem]
    rw [secondState_mem _ _ _ _ _ _ _ _ hN hsep,
      Execution.firstStreamState_mem _ _ _ _ _ _ _ hN]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.InputPrefix
