import bench.tritonbench_g.layer_norm_fwd.Proofs.AffineSecond
import bench.tritonbench_g.layer_norm_fwd.Proofs.AffineSpecification
import bench.tritonbench_g.layer_norm_fwd.Proofs.NormPrefix

/-! Successful execution and independent full tagged-memory contract for the
complete original source. All ten flags, both seven-round signed-seed streams,
both reductions and all seven output paths remain in the executed program. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.RealExecution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def firstOutputState (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (eps p : ℝ) (u : BlockState) : BlockState :=
  Affine.firstState f r (u.pid * stride.y) N block (InputPrefix.values f r stride M N p u)
    (Formula.mean f (snapshot r stride u) M N p u.pid)
    (Formula.rstd f (snapshot r stride u) M N eps p u.pid)
    (NormPrefix.state f r stride M N block eps p u)

def state (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (eps p : ℝ) (u : BlockState) : BlockState :=
  Affine.state f r (u.pid * stride.y) (u.pid * stride.y1) N block
    (InputPrefix.values f r stride M N p u)
    (Formula.mean f (snapshot r stride u) M N p u.pid)
    (Formula.rstd f (snapshot r stride u) M N eps p u.pid)
    (NormPrefix.state f r stride M N block eps p u)

theorem source_exec (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    exec (Execution.source f r stride M N eps p block hBlock) u =
      some (state f r stride M N block eps p u) := by
  unfold exec
  conv_lhs => rw [← List.take_append_drop 18 (Execution.source f r stride M N eps p block hBlock).body]
  rw [ProgramRewrite.append, NormPrefix.source_exec _ _ _ _ _ _ _ _ _ _ hN hsep,
    Option.bind_some, Affine.source_phase]
  apply Affine.phase_exec
  · rw [NormPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide),
      InputPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact (Execution.loadState_pointers f r stride N block u).1
  · intro hf
    rw [NormPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide),
      InputPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact (Execution.loadState_pointers f r stride N block u).2.2.2.2 hf
  · rw [NormPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide),
      InputPrefix.state_regs _ _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact Execution.loadState_cols f r stride N block u
  · exact NormPrefix.state_x f r stride M N block eps p u hN hsep
  · exact NormPrefix.state_mean f r stride M N block eps p u
  · exact NormPrefix.state_rstd f r stride M N block eps p u

theorem firstOutputState_readsUnchanged (f : Flags) (r : Regions) (stride : Strides)
    (M N block : Nat) (eps p : ℝ) (u : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) :
    Domain.ReadsUnchanged f r (firstOutputState f r stride M N block eps p u) u :=
  Domain.readsUnchanged_trans f r _ _ _
    (Affine.firstState_readsUnchanged f r hsep _ _ _ _ _ _ _ hN)
    (NormPrefix.state_readsUnchanged f r stride M N block eps p u hN hsep)

theorem firstOutputState_mem (f : Flags) (r : Regions) (stride : Strides)
    (M N block : Nat) (eps p : ℝ) (u : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) (region : RegionName) (offset : Nat) :
    (firstOutputState f r stride M N block eps p u).mem region offset =
      if region = r.y ∧ Contract.inRow (u.pid * stride.y) N offset then
        MemCell.real (Formula.y f (snapshot r stride u) M N eps p u.pid (offset-u.pid*stride.y))
      else (NormPrefix.state f r stride M N block eps p u).mem region offset := by
  rw [firstOutputState, Affine.firstState_mem _ _ _ _ _ _ _ _ _ hN]
  split_ifs with h
  · have hcol : offset - u.pid * stride.y < N := by
      have hin := h.2
      unfold Contract.inRow at hin
      omega
    exact congrArg MemCell.real (Affine.affineValues_y f r stride M N eps p u.pid
      (offset-u.pid*stride.y) (NormPrefix.state f r stride M N block eps p u) u
      (NormPrefix.state_readsUnchanged f r stride M N block eps p u hN hsep) hcol)
  · rfl

theorem state_mem (f : Flags) (r : Regions) (stride : Strides)
    (M N block : Nat) (eps p : ℝ) (u : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) (region : RegionName) (offset : Nat) :
    (state f r stride M N block eps p u).mem region offset =
      if f.w1 && decide (region = r.y1 ∧ Contract.inRow (u.pid * stride.y1) N offset) then
        MemCell.real (Formula.y1 f (snapshot r stride u) M N eps p u.pid (offset-u.pid*stride.y1))
      else if region = r.y ∧ Contract.inRow (u.pid * stride.y) N offset then
        MemCell.real (Formula.y f (snapshot r stride u) M N eps p u.pid (offset-u.pid*stride.y))
      else (NormPrefix.state f r stride M N block eps p u).mem region offset := by
  change (Affine.secondState f r (u.pid*stride.y1) N block
    (Affine.mathematicalXhat f (snapshot r stride u) M N eps p u.pid)
    (firstOutputState f r stride M N block eps p u)).mem region offset = _
  rw [Affine.secondState_mem _ _ _ _ _ _ _ hN]
  by_cases hstore : (f.w1 && decide (region = r.y1 ∧
    Contract.inRow (u.pid * stride.y1) N offset)) = Bool.true
  · simp only [hstore, ↓reduceIte]
    have hparts : f.w1 = Bool.true ∧
        (region = r.y1 ∧ Contract.inRow (u.pid * stride.y1) N offset) := by
      simpa only [Bool.and_eq_true, decide_eq_true_eq] using hstore
    have hcol : offset - u.pid * stride.y1 < N := by
      have hin := hparts.2.2
      unfold Contract.inRow at hin
      omega
    exact congrArg MemCell.real (Affine.affineValues_y1 f r stride M N eps p u.pid
      (offset-u.pid*stride.y1) (firstOutputState f r stride M N block eps p u) u
      (firstOutputState_readsUnchanged f r stride M N block eps p u hN hsep) hcol hparts.1)
  · simp only [hstore, Bool.false_eq_true, if_false]
    exact firstOutputState_mem f r stride M N block eps p u hN hsep region offset

/-- All seven outputs and every other tagged memory cell match the independent
initial-snapshot contract, including output aliases through explicit store order. -/
theorem source_post (f : Flags) (r : Regions) (stride : Strides)
    (M N block : Nat) (eps p : ℝ) (u : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) :
    Contract.Post f r stride M N eps p u (state f r stride M N block eps p u) := by
  intro region offset
  rw [state_mem _ _ _ _ _ _ _ _ _ hN hsep, NormPrefix.state_mem,
    InputPrefix.state_mem _ _ _ _ _ _ _ _ hN hsep]
  unfold Contract.expectedCell
  split_ifs <;> rfl

theorem source_correct (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    ∃ final, exec (Execution.source f r stride M N eps p block hBlock) u = some final ∧
      Contract.Post f r stride M N eps p u final :=
  ⟨state f r stride M N block eps p u, source_exec f r stride M N eps p block hBlock u hN hsep,
    source_post f r stride M N block eps p u hN hsep⟩

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.RealExecution
