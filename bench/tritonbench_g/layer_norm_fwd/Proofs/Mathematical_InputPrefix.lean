import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_Residual
import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_InputPreservation

/-! Composition of the actual first fifteen source statements. Every load
and random operation is executed. The resulting combined input is calculated
from the initial memory snapshot, under explicit active-region separation. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.InputPrefix
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def firstValues (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) : Fin block → ℝ :=
  fun j => Formula.stream f (Formula.paddedInputs (snapshot r stride u) N)
    M p Bool.false u.pid j.val

def twoValues (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) : Fin block → ℝ :=
  fun j => if f.x1 then firstValues f r stride M N block p u j +
    Formula.stream f (Formula.paddedInputs (snapshot r stride u) N)
      M p Bool.true u.pid j.val else firstValues f r stride M N block p u j

def secondState (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) : BlockState :=
  if f.x1 then InputSuffix.bodyState f r M u.pid (u.pid * stride.x1) N block p
    (firstValues f r stride M N block p u) (Execution.firstStreamState f r stride N block p u)
  else Execution.firstStreamState f r stride N block p u

theorem secondStmt_exec (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) (hBlock : Random.ArangeSizeDomain block) :
    stepStmt (.ifThen (.constBool f.x1) (InputSuffix.secondBody f r M N block p))
      (Execution.firstStreamState f r stride N block p u) =
        some (secondState f r stride M N block p u) := by
  rw [InputSuffix.step_if_const]
  cases hf : f.x1
  · simp [secondState, hf]
  · simp only [↓reduceIte, secondState, hf]
    apply InputSuffix.secondBody_exec _ _ _ _ _ _ _ _ _ _ hBlock
    · rw [Execution.firstStreamState_regs _ _ _ _ _ _ _ _ _ _
        (by decide) (by decide) (by decide)]
      exact (Execution.loadState_pointers f r stride N block u).2.1 hf
    · exact Execution.firstStreamState_row f r stride N block p u
    · exact Execution.firstStreamState_cols f r stride N block p u
    · exact Execution.firstStreamState_x f r stride M N block p u

theorem secondState_regs (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hx : name ≠ "x") (hx1 : name ≠ "x1")
    (hrs : name ≠ "rowscale") (hkeep : name ≠ "keep_mask") :
    (secondState f r stride M N block p u).regs dtype shape name =
      (Execution.loadState f r stride N block u).regs dtype shape name := by
  cases hf : f.x1
  · simp only [secondState, hf, Bool.false_eq_true, if_false]
    exact Execution.firstStreamState_regs f r stride N block p u dtype shape name hx hrs hkeep
  · simp only [secondState, hf, ↓reduceIte]
    rw [InputSuffix.bodyState_regs _ _ _ _ _ _ _ _ _ _ _ _ _ hx hx1 hrs hkeep]
    exact Execution.firstStreamState_regs f r stride N block p u dtype shape name hx hrs hkeep

theorem secondState_readsUnchanged (f : Flags) (r : Regions) (stride : Strides)
    (M N block : Nat) (p : ℝ) (u : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) :
    Domain.ReadsUnchanged f r (secondState f r stride M N block p u) u := by
  have hfirst := Execution.firstStreamState_readsUnchanged f r stride N block p u hN hsep
  cases hf : f.x1
  · simpa [secondState, hf] using hfirst
  · simp only [secondState, hf, ↓reduceIte]
    exact Domain.readsUnchanged_trans f r _ _ _
      (InputSuffix.bodyState_readsUnchanged f r M u.pid (u.pid * stride.x1) N block p
        (firstValues f r stride M N block p u)
        (Execution.firstStreamState f r stride N block p u) hN hsep) hfirst

theorem secondState_x (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    (secondState f r stride M N block p u).regs .real [block] "x" =
      some (Execution.realTile (twoValues f r stride M N block p u)) := by
  cases hf : f.x1
  · simp only [secondState, hf, Bool.false_eq_true, if_false]
    rw [Execution.firstStreamState_x f r stride M N block p u]
    congr 1
    ext j
    simp [Execution.realTile, twoValues, firstValues, hf]
  · simp only [secondState, hf, ↓reduceIte, InputSuffix.bodyState_x]
    congr 1
    ext j
    simp only [Execution.realTile, twoValues, hf, ↓reduceIte]
    rw [InputSuffix.streamValues_snapshot f r stride M u.pid N block p
      (Execution.firstStreamState f r stride N block p u) u
      (Execution.firstStreamState_readsUnchanged f r stride N block p u hN hsep) hf]

def afterResidual (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) : BlockState :=
  InputSuffix.residualState f.residual r.residual (u.pid * stride.residual) N block
    (twoValues f r stride M N block p u) (secondState f r stride M N block p u)

theorem afterResidual_regs (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hx : name ≠ "x") (hx1 : name ≠ "x1") (hrs : name ≠ "rowscale")
    (hkeep : name ≠ "keep_mask") (hr : name ≠ "residual") :
    (afterResidual f r stride M N block p u).regs dtype shape name =
      (Execution.loadState f r stride N block u).regs dtype shape name := by
  rw [afterResidual, InputSuffix.residualState_regs _ _ _ _ _ _ _ _ _ _ hx hr]
  exact secondState_regs f r stride M N block p u dtype shape name hx hx1 hrs hkeep

theorem afterResidual_readsUnchanged (f : Flags) (r : Regions) (stride : Strides)
    (M N block : Nat) (p : ℝ) (u : BlockState) (hN : N ≤ block)
    (hsep : Domain.ReadWriteSeparated f r) :
    Domain.ReadsUnchanged f r (afterResidual f r stride M N block p u) u := by
  apply Domain.readsUnchanged_trans f r _ (secondState f r stride M N block p u) u
  · apply Domain.readsUnchanged_of_mem
    exact InputSuffix.residualState_mem _ _ _ _ _ _ _
  · exact secondState_readsUnchanged f r stride M N block p u hN hsep

theorem afterResidual_x (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    (afterResidual f r stride M N block p u).regs .real [block] "x" =
      some (Reduction.combinedTile f (snapshot r stride u) M N block p u.pid) := by
  rw [afterResidual, InputSuffix.residualState_x _ _ _ _ _ _ _
    (secondState_x f r stride M N block p u hN hsep)]
  congr 1
  ext j
  simp only [Execution.realTile, Reduction.combinedTile, Formula.combined, twoValues, firstValues]
  cases hr : f.residual
  · simp [hr]
  · simp only [hr, ↓reduceIte]
    rw [InputSuffix.rowValues_residual_snapshot f r stride u.pid N block
      (secondState f r stride M N block p u) u
      (secondState_readsUnchanged f r stride M N block p u hN hsep) hr]

def values (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (p : ℝ) (u : BlockState) : Nat → ℝ :=
  Formula.combined f (Formula.paddedInputs (snapshot r stride u) N) M p u.pid

def state (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) : BlockState :=
  InputSuffix.residualOutState f.residualOut r.residualOut (u.pid * stride.residualOut) N block
    (values f r stride M N p u) (afterResidual f r stride M N block p u)

theorem state_regs (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hx : name ≠ "x") (hx1 : name ≠ "x1") (hrs : name ≠ "rowscale")
    (hkeep : name ≠ "keep_mask") (hr : name ≠ "residual") :
    (state f r stride M N block p u).regs dtype shape name =
      (Execution.loadState f r stride N block u).regs dtype shape name := by
  rw [state, InputSuffix.residualOutState_regs]
  exact afterResidual_regs f r stride M N block p u dtype shape name hx hx1 hrs hkeep hr

theorem state_x (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    (state f r stride M N block p u).regs .real [block] "x" =
      some (Reduction.combinedTile f (snapshot r stride u) M N block p u.pid) := by
  rw [state, InputSuffix.residualOutState_regs]
  exact afterResidual_x f r stride M N block p u hN hsep

theorem phase_exec (f : Flags) (r : Regions) (stride : Strides) (M N block : Nat)
    (p : ℝ) (u : BlockState) (hBlock : Random.ArangeSizeDomain block)
    (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    stepStmts (InputSuffix.phase f r M N block p)
      (Execution.firstStreamState f r stride N block p u) =
        some (state f r stride M N block p u) := by
  have hp := Execution.loadState_pointers f r stride N block u
  have hres := InputSuffix.residualStmt_exec f.residual r.residual
    (u.pid * stride.residual) N block (twoValues f r stride M N block p u)
    (secondState f r stride M N block p u)
    (by intro hf
        rw [secondState_regs _ _ _ _ _ _ _ _ _ _ _
          (by decide) (by decide) (by decide) (by decide)]
        exact hp.2.2.1 hf)
    (by rw [secondState_regs _ _ _ _ _ _ _ _ _ _ _
          (by decide) (by decide) (by decide) (by decide)]
        exact Execution.loadState_cols f r stride N block u)
    (secondState_x f r stride M N block p u hN hsep)
  have hout := InputSuffix.residualOutStmt_exec f.residualOut r.residualOut
    (u.pid * stride.residualOut) N block (values f r stride M N p u)
    (afterResidual f r stride M N block p u)
    (by intro hf
        rw [afterResidual_regs _ _ _ _ _ _ _ _ _ _ _
          (by decide) (by decide) (by decide) (by decide) (by decide)]
        exact hp.2.2.2.1 hf)
    (by rw [afterResidual_regs _ _ _ _ _ _ _ _ _ _ _
          (by decide) (by decide) (by decide) (by decide) (by decide)]
        exact Execution.loadState_cols f r stride N block u)
    (afterResidual_x f r stride M N block p u hN hsep)
  rw [InputSuffix.phase, stepStmts.cons_some (secondStmt_exec f r stride M N block p u hBlock),
    stepStmts.cons_some hres, InputSuffix.exec_single]
  exact hout

/-- Successful execution of the actual full input/residual prefix, including
both complete RNG streams and every conditional mask/residual output store. -/
theorem source_exec (f : Flags) (r : Regions) (stride : Strides)
    (M N : Nat) (eps p : ℝ) (block : Nat) (hBlock : Random.ArangeSizeDomain block)
    (u : BlockState) (hN : N ≤ block) (hsep : Domain.ReadWriteSeparated f r) :
    stepStmts ((Execution.source f r stride M N eps p block hBlock).body.take 15) u =
      some (state f r stride M N block p u) := by
  change stepStmts ((Execution.source f r stride M N eps p block hBlock).body.take (12+3)) u = _
  rw [List.take_add, ProgramRewrite.append, Execution.source_firstStream_exec,
    Option.bind_some, InputSuffix.source_phase]
  exact phase_exec f r stride M N block p u hBlock hN hsep

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.InputPrefix
