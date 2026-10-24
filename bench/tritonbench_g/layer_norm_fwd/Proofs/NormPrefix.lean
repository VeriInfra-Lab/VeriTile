import bench.tritonbench_g.layer_norm_fwd.Proofs.InputPrefixMemory

/-! Actual source execution through both reductions and Rstd storage. The
normalization phase receives its combined tile from executed input statements,
not from an assumed or precomputed intermediate value. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.NormPrefix
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def state (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (eps p : ℝ) (u : BlockState) : BlockState :=
  Normalization.state f (snapshot r stride u) r M N block eps p u.pid
    (InputPrefix.state f r stride M N block p u)

theorem source_exec (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    stepStmts ((Execution.source f r stride M N eps p block hBlock).body.take 18) u =
      some (state f r stride M N block eps p u) := by
  change stepStmts ((Execution.source f r stride M N eps p block hBlock).body.take (15+3)) u = _
  rw [List.take_add, ProgramRewrite.append,
    InputPrefix.source_exec _ _ _ _ _ _ _ _ _ _ hN hsep, Option.bind_some]
  apply Normalization.source_phase_exec _ _ _ _ _ _ _ _ _ _ _ hN
  · exact InputPrefix.state_x f r stride M N block p u hN hsep
  · rw [InputPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _
      (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact Execution.loadState_cols f r stride N block u
  · rw [InputPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _
      (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact Execution.loadState_row f r stride N block u

theorem state_regs (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (eps p : ℝ) (u : BlockState) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hm : name ≠ "mean") (hx : name ≠ "xbar")
    (hv : name ≠ "var") (hs : name ≠ "rstd") :
    (state f r stride M N block eps p u).regs dtype shape name =
      (InputPrefix.state f r stride M N block p u).regs dtype shape name := by
  cases hr : f.rms
  all_goals simp [state, Normalization.state, Normalization.statisticsState,
    Normalization.centerVarianceState, Normalization.meanState, Memory.writeCell,
    hr, hm, hx, hv, hs]

theorem state_x (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (eps p : ℝ) (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    (state f r stride M N block eps p u).regs .real [block] "x" =
      some (Reduction.combinedTile f (snapshot r stride u) M N block p u.pid) := by
  rw [state, Normalization.state_x]
  exact InputPrefix.state_x f r stride M N block p u hN hsep

theorem state_mean (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (eps p : ℝ) (u : BlockState) (hr : f.rms = Bool.false) :
    (state f r stride M N block eps p u).regs .real [] "mean" =
      some (Tile.scalar (some (Formula.mean f (snapshot r stride u) M N p u.pid))) :=
  Normalization.state_mean _ _ _ _ _ _ _ _ _ _ hr

theorem state_rstd (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (eps p : ℝ) (u : BlockState) :
    (state f r stride M N block eps p u).regs .real [] "rstd" =
      some (Tile.scalar (some (Formula.rstd f (snapshot r stride u) M N eps p u.pid))) :=
  Normalization.state_rstd _ _ _ _ _ _ _ _ _ _

theorem state_mem (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (eps p : ℝ) (u : BlockState) (region : RegionName) (offset : Nat) :
    (state f r stride M N block eps p u).mem region offset =
      if region = r.rstd ∧ offset = u.pid then
        MemCell.real (Formula.rstd f (snapshot r stride u) M N eps p u.pid)
      else if !f.rms && decide (region = r.mean ∧ offset = u.pid) then
        MemCell.real (Formula.mean f (snapshot r stride u) M N p u.pid)
      else (InputPrefix.state f r stride M N block p u).mem region offset :=
  Normalization.state_mem _ _ _ _ _ _ _ _ _ _ _ _

theorem state_readsUnchanged (f : Flags) (r : Regions) (stride : Strides)
    (M N block : Nat) (eps p : ℝ) (u : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) :
    Domain.ReadsUnchanged f r (state f r stride M N block eps p u) u := by
  intro region hr offset
  rw [state_mem]
  have hstd : region ≠ r.rstd := hsep region hr r.rstd (by simp [Domain.writeRegion])
  simp only [hstd, false_and, if_false]
  have hinput := InputPrefix.state_readsUnchanged f r stride M N block p u hN hsep region hr offset
  cases hrms : f.rms
  · have hm : region ≠ r.mean := hsep region hr r.mean (by simp [Domain.writeRegion, hrms])
    simpa [hrms, hm] using hinput
  · simpa [hrms] using hinput

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.NormPrefix
