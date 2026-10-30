import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_FirstStream
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_SecondStream
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_InputDomain

/-! Registers and active input memory survive the complete first and second
input streams. Later source loads are identified with the initial snapshot
using an explicit read/write alias domain. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

namespace Execution

theorem loadState_pointers (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (u : BlockState) :
    (loadState f r stride N block u).regs .ptr [] "Y" =
      some (Tile.scalar (r.y, u.pid * stride.y)) ∧
    (f.x1 = Bool.true → (loadState f r stride N block u).regs .ptr [] "X1" =
      some (Tile.scalar (r.x1, u.pid * stride.x1))) ∧
    (f.residual = Bool.true → (loadState f r stride N block u).regs .ptr [] "RESIDUAL" =
      some (Tile.scalar (r.residual, u.pid * stride.residual))) ∧
    (f.residualOut = Bool.true → (loadState f r stride N block u).regs .ptr [] "RESIDUAL_OUT" =
      some (Tile.scalar (r.residualOut, u.pid * stride.residualOut))) ∧
    (f.w1 = Bool.true → (loadState f r stride N block u).regs .ptr [] "Y1" =
      some (Tile.scalar (r.y1, u.pid * stride.y1))) := by
  rcases f with ⟨rms, residual, residualOut, bias, dropout, dropoutMask, rowscale, x1, w1, b1⟩
  cases residual <;> cases residualOut <;> cases x1 <;> cases w1
  all_goals simp [loadState, ptrState]

theorem firstStreamState_regs (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (p : ℝ) (u : BlockState) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hx : name ≠ "x") (hs : name ≠ "rowscale") (hk : name ≠ "keep_mask") :
    (firstStreamState f r stride N block p u).regs dtype shape name =
      (loadState f r stride N block u).regs dtype shape name := by
  cases hscale : f.rowscale <;> cases hdrop : f.dropout <;> cases hmask : f.dropoutMask
  all_goals simp [firstStreamState, firstScaledState, scaleState,
    dropoutState, dropoutBase, hscale, hdrop, hmask, hx, hs, hk]

theorem firstStreamState_row (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (p : ℝ) (u : BlockState) :
    (firstStreamState f r stride N block p u).regs .nat [] "row" =
      some (Tile.scalar u.pid) := by
  rw [firstStreamState_regs _ _ _ _ _ _ _ _ _ _ (by decide) (by decide) (by decide)]
  exact loadState_row f r stride N block u

theorem firstStreamState_cols (f : Flags) (r : Regions) (stride : Strides) (N block : Nat)
    (p : ℝ) (u : BlockState) :
    (firstStreamState f r stride N block p u).regs .nat [block] "cols" =
      some (Tile.vec (fun j => j.val)) := by
  rw [firstStreamState_regs _ _ _ _ _ _ _ _ _ _ (by decide) (by decide) (by decide)]
  exact loadState_cols f r stride N block u

theorem firstStreamState_readsUnchanged (f : Flags) (r : Regions) (stride : Strides)
    (N block : Nat) (p : ℝ) (u : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) :
    Domain.ReadsUnchanged f r (firstStreamState f r stride N block p u) u := by
  intro region hr offset
  rw [firstStreamState_mem _ _ _ _ _ _ _ hN]
  cases hd : f.dropout <;> cases hm : f.dropoutMask
  all_goals simp only [hd, hm, Bool.false_and, Bool.true_and, Bool.false_eq_true, if_false]
  have hw : Domain.writeRegion f r r.dropoutMask := by
    simp [Domain.writeRegion, hd, hm]
  simp [hsep region hr r.dropoutMask hw]

end Execution

namespace InputSuffix

theorem bodyState_regs (f : Flags) (r : Regions) (M row start N block : Nat)
    (p : ℝ) (x : Fin block → ℝ) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hx : name ≠ "x") (hx1 : name ≠ "x1")
    (hrs : name ≠ "rowscale") (hkeep : name ≠ "keep_mask") :
    (bodyState f r M row start N block p x s).regs dtype shape name =
      s.regs dtype shape name := by
  rw [bodyState, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hx]
  exact droppedState_regs f r M row start N block p s dtype shape name hx1 hrs hkeep

theorem bodyState_readsUnchanged (f : Flags) (r : Regions) (M row start N block : Nat)
    (p : ℝ) (x : Fin block → ℝ) (s : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) :
    Domain.ReadsUnchanged f r (bodyState f r M row start N block p x s) s := by
  intro region hr offset
  rw [bodyState_mem _ _ _ _ _ _ _ _ _ _ hN]
  cases hd : f.dropout <;> cases hm : f.dropoutMask
  all_goals simp only [hd, hm, Bool.false_and, Bool.true_and, Bool.false_eq_true, if_false]
  have hw : Domain.writeRegion f r r.dropoutMask := by
    simp [Domain.writeRegion, hd, hm]
  simp [hsep region hr r.dropoutMask hw]

/-- The actual second stream uses the first stream's final memory. Separation
identifies all active loads, including the signed seed, with the original
snapshot without imposing conditions on inactive input buffers. -/
theorem streamValues_snapshot (f : Flags) (r : Regions) (stride : Strides)
    (M row N block : Nat) (p : ℝ) (s u : BlockState)
    (hread : Domain.ReadsUnchanged f r s u) (hx1 : f.x1 = Bool.true) (j : Fin block) :
    streamValues f r M row (row * stride.x1) N block p s j =
      Formula.stream f (Formula.paddedInputs (snapshot r stride u) N) M p Bool.true row j.val := by
  have hx (offset : Nat) : s.readMem r.x1 offset = u.readMem r.x1 offset :=
    Domain.readsUnchanged_readMem f r s u hread r.x1
      (by simp [Domain.readRegion, hx1]) offset
  have hrs (hs : f.rowscale = Bool.true) (offset : Nat) :
      s.readMem r.rowscale offset = u.readMem r.rowscale offset :=
    Domain.readsUnchanged_readMem f r s u hread r.rowscale
      (by simp [Domain.readRegion, hs]) offset
  have hseed (hd : f.dropout = Bool.true) (offset : Nat) :
      s.readMemValue .int r.seeds offset = u.readMemValue .int r.seeds offset :=
    Domain.readsUnchanged_seed f r s u hread hd offset
  cases hs : f.rowscale <;> cases hd : f.dropout
  all_goals simp [streamValues, scaledValues, rowValues, Formula.stream,
    Formula.paddedInputs, snapshot, hs, hd, hx, hrs, hseed]

theorem rowValues_residual_snapshot (f : Flags) (r : Regions) (stride : Strides)
    (row N block : Nat) (s u : BlockState)
    (hread : Domain.ReadsUnchanged f r s u) (hr : f.residual = Bool.true) (j : Fin block) :
    rowValues r.residual (row * stride.residual) N block s j =
      (Formula.paddedInputs (snapshot r stride u) N).residual row j.val := by
  have heq := Domain.readsUnchanged_readMem f r s u hread r.residual
    (by simp [Domain.readRegion, hr]) (row * stride.residual + j.val)
  simp only [rowValues, Formula.paddedInputs, snapshot, heq]

end InputSuffix
end
end VeriTile.Bench.TritonBenchG.LayerNormFwd
