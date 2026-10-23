import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.Kernel

namespace VeriTile.Bench.Repairs.DestindexCopyKv2.RawGrid
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- Execute the authoritative kv2 quote once for each supplied raw program ID. -/
def run (rest : Algebra α) (c : Config) : List Nat → State α → Option (State α)
  | [], s => some s
  | p::ps, s => (FP.Structural.exec (Tile.rawBackend rest) (kernel (layout c))
      {s with pids := fun d => if d=0 then p else s.pids d}).bind (run rest c ps)

theorem runs (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α) :
    run rest c ps s=some (Mathematical.run rest c ps s) := by
  induction ps generalizing s with
  | nil => rfl
  | cons p ps ih =>
    rw [run,(Tile.execution rest (layout c) {s with pids := fun d => if d=0 then p else s.pids d}).property]
    exact ih _

theorem other_frame (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α)
    (r : RegionName) (x : Nat) (outside : r≠DestindexCopyKv1.outputRegion) :
    (Mathematical.run rest c ps s).mem r x=s.mem r x := by
  induction ps generalizing s with
  | nil => rfl
  | cons p ps ih =>
    change (Mathematical.run rest c ps (Mathematical.step rest c s p)).mem r x=s.mem r x
    rw [ih]
    exact Tile.frame rest (layout c) _ r x (Or.inl outside)

/-- Full raw row-copy result. The width equation is the exact condition making
the source's omitted feature mask valid; no floating arithmetic law is used. -/
theorem complete (rest : Algebra α) (c : Config) (width : block c.dims=c.dims)
    (ps : List Nat) (before : State α) (source : Nat → α) (dest : Nat → Nat)
    (bound : DestindexCopyKv1.Grid.Bound before source dest)
    (unique : ∀ p q, p<c.count → q<c.count → dest p=dest q → p=q)
    (schedule : ∀ p,p∈ps ↔ p<c.count) :
    ∃ final, run rest c ps before=some final ∧
      (∀ row head dim,row<c.count → head<c.heads → dim<c.dims →
        final.mem DestindexCopyKv1.outputRegion (flat c (dest row) head dim)=
          .mk .real (source (flat c row head dim))) ∧
      (∀ r x,r≠DestindexCopyKv1.outputRegion ∨ ¬touched c dest x → final.mem r x=before.mem r x) := by
  refine ⟨Mathematical.run rest c ps before,runs rest c ps before,?_,?_⟩
  · intro row head dim hr hh hd
    rw [Mathematical.run_memory rest c width ps before before rfl]
    exact DestindexCopyKv1.Grid.run_read rest c ps before source dest bound unique
      (fun p hp => (schedule p).mp hp) row head dim ((schedule row).mpr hr) hh hd
  · intro r x outside
    by_cases other : r≠DestindexCopyKv1.outputRegion
    · exact other_frame rest c ps before r x other
    · have same : r=DestindexCopyKv1.outputRegion := Classical.byContradiction other
      subst r
      rw [Mathematical.run_memory rest c width ps before before rfl]
      apply DestindexCopyKv1.Grid.run_frame rest c ps before source dest bound x
      intro p hp head dim hh hd equal
      exact (outside.resolve_left (by simp)) ⟨p,head,dim,(schedule p).mp hp,hh,hd,equal⟩

#axiomsClean complete
end
end VeriTile.Bench.Repairs.DestindexCopyKv2.RawGrid
