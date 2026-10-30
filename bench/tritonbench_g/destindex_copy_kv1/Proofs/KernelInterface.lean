import bench.tritonbench_g.destindex_copy_kv1.Proofs.Grid

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.KernelInterface
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- The actual complete source executes in the supplied program order. -/
inductive Run (rest : Algebra α) (c : Config) : List Nat → State α → State α → Prop where
  | nil (s) : Run rest c [] s s
  | cons {pid ps s middle final}
      (step : FP.Structural.exec (Tile.rawBackend rest) (kernel (layout c))
        {s with pids := fun d => if d=0 then pid else s.pids d}=some middle)
      (tail : Run rest c ps middle final) : Run rest c (pid::ps) s final

theorem run (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α) :
    Run rest c ps s (Grid.run rest c ps s) := by
  induction ps generalizing s with
  | nil => exact Run.nil s
  | cons pid ps ih =>
    exact Run.cons (Tile.execution rest (layout c)
      {s with pids := fun d => if d=0 then pid else s.pids d}).property (ih _)

theorem other_frame (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (r : RegionName) (o : Nat) (outside : r≠outputRegion) :
    (Grid.run rest c ps s).mem r o=s.mem r o := by
  induction ps generalizing s with
  | nil => rfl
  | cons pid ps ih =>
    change (Grid.run rest c ps (Grid.step rest c s pid)).mem r o=s.mem r o
    rw [ih]
    exact Tile.frame rest (layout c) _ r o (Or.inl outside)

/-- Complete launch, independent row-copy values, and every untouched typed
memory cell. Inputs are raw arrays, geometry and source memory reads. -/
theorem complete (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (source : Nat → α) (dest : Nat → Nat) (reads : Grid.Bound s source dest)
    (unique : ∀ p q,p<c.count → q<c.count → dest p=dest q → p=q)
    (order : ∀ p,p∈ps ↔ p<c.count) :
    ∃ final, Run rest c ps s final ∧
      RowCopy c dest source (fun x => (s.mem outputRegion x).read .real)
        (fun x => (final.mem outputRegion x).read .real) ∧
      (∀ r o,(r≠outputRegion ∨ ∀ p,p<c.count → ∀ head dim,
        head<c.heads → dim<c.dims → o≠flat c (dest p) head dim) →
        final.mem r o=s.mem r o) := by
  refine ⟨Grid.run rest c ps s,run rest c ps s,
    Grid.correct rest c ps s source dest reads unique order,?_⟩
  intro r o outside
  by_cases eq : r=outputRegion
  · subst r
    apply Grid.run_frame rest c ps s source dest reads
    intro p hp head dim hh hd
    exact outside.resolve_left (by simp) p ((order p).mp hp) head dim hh hd
  · exact other_frame rest c ps s r o eq

#axiomsClean complete
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.KernelInterface
