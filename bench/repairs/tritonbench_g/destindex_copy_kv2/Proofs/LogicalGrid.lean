import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.SourceAgreement
import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.SourceSafety
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.LogicalGrid

/-! Reuse the finite logical copy-grid proof after proving that each actual
kv2 source step has the same memory effect. The omitted feature mask is
justified by the explicit width equality, never by a modified source. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv2.LogicalGrid
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
variable {α : Type} [Inhabited α]

abbrev Bound := @DestindexCopyKv1.LogicalGrid.Bound
abbrev launch := @DestindexCopyKv1.LogicalGrid.launch

def step (rest : Algebra α) (c : Config) (s : State α) (pid : Nat) : State α :=
  (Tile.execution rest (layout c) (launch pid s)).val

def run (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α) : State α :=
  ps.foldl (step rest c) s

def runSource (rest : Algebra α) (c : Config) : List Nat → State α → Option (State α)
  | [], s => some s
  | pid :: ps, s => (exec (Tile.rawBackend rest) (kernel (layout c)) (launch pid s)).bind
      (runSource rest c ps)

theorem source_run (rest : Algebra α) (c : Config) (ps : List Nat) (s : State α) :
    runSource rest c ps s = some (run rest c ps s) := by
  induction ps generalizing s with
  | nil => rfl
  | cons pid ps ih =>
      rw [runSource, (Tile.execution rest (layout c) (launch pid s)).property]
      exact ih _

theorem step_memory (rest : Algebra α) (c : Config) (width : block c.dims = c.dims)
    (s t : State α) (same : s.mem = t.mem) (pid : Nat) :
    (step rest c s pid).mem = (DestindexCopyKv1.LogicalGrid.step rest c t pid).mem := by
  unfold step
  rw [same_source_memory rest _ width.symm]
  unfold DestindexCopyKv1.LogicalGrid.step
  rw [DestindexCopyKv1.Tile.execution_memory, DestindexCopyKv1.Tile.execution_memory]
  unfold DestindexCopyKv1.Tile.copied
  rw [VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_memory,
    VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_memory]
  simp only [DestindexCopyKv1.Tile.loaded, DestindexCopyKv1.Tile.inputOffset,
    DestindexCopyKv1.Tile.outputOffset, DestindexCopyKv1.Tile.destination, launch, DestindexCopyKv1.LogicalGrid.launch, same]
  rfl

theorem run_memory (rest : Algebra α) (c : Config) (width : block c.dims = c.dims)
    (ps : List Nat) (s t : State α) (same : s.mem = t.mem) :
    (run rest c ps s).mem = (DestindexCopyKv1.LogicalGrid.run rest c ps t).mem := by
  induction ps generalizing s t with
  | nil => exact same
  | cons pid ps ih => exact ih _ _ (step_memory rest c width s t same pid)

theorem step_bound (rest : Algebra α) (c : Config) (width : block c.dims = c.dims)
    (s : State α) (pid : Nat) (source : Nat → Nat → Nat → α) (dest : Nat → Nat)
    (h : Bound c s source dest) : Bound c (step rest c s pid) source dest := by
  have hh := DestindexCopyKv1.LogicalGrid.step_bound rest c s pid source dest h
  constructor
  · intro row hr head dim hh' hd
    rw [step_memory rest c width s s rfl pid]
    exact hh.input row hr head dim hh' hd
  · intro row hr
    rw [step_memory rest c width s s rfl pid]
    exact hh.metadata row hr

theorem offset (c : Config) (s : State α) (pid : Nat) (hp : pid < c.count)
    (source : Nat → Nat → Nat → α) (dest : Nat → Nat) (h : Bound c s source dest)
    (i : TileIndex [block c.heads,block c.dims]) :
    Tile.outputOffset (layout c) (launch pid s) i = flat c (dest pid) i.1.val i.2.1.val :=
  DestindexCopyKv1.LogicalGrid.offset c s pid hp source dest h i

/-- Full output and frame transfer; no new value-grid induction is needed. -/
theorem complete (rest : Algebra α) (c : Config) (width : block c.dims = c.dims)
    (ps : List Nat) (s : State α) (source : Nat → Nat → Nat → α) (dest : Nat → Nat)
    (h : Bound c s source dest)
    (unique : ∀ p q, p<c.count → q<c.count → dest p=dest q → p=q)
    (schedule : ∀ p, p∈ps ↔ p<c.count) :
    ∃ final, runSource rest c ps s = some final ∧
      (∀ row head dim, row<c.count → head<c.heads → dim<c.dims →
        final.mem DestindexCopyKv1.outputRegion (flat c (dest row) head dim) =
          .mk .real (source row head dim)) ∧
      (∀ r o, r≠DestindexCopyKv1.outputRegion ∨
        (∀ row head dim, row<c.count → head<c.heads → dim<c.dims →
          o≠flat c (dest row) head dim) → final.mem r o=s.mem r o) := by
  refine ⟨run rest c ps s, source_run rest c ps s, ?_, ?_⟩
  · intro row head dim hr hh hd
    rw [run_memory rest c width ps s s rfl]
    exact DestindexCopyKv1.LogicalGrid.run_read rest c ps s source dest h unique
      (fun p hp => (schedule p).mp hp) row head dim ((schedule row).mpr hr) hh hd
  · intro r o outside
    rw [run_memory rest c width ps s s rfl]
    rcases outside with hr | ho
    · exact DestindexCopyKv1.LogicalGrid.other_frame rest c ps s r o hr
    · by_cases hr : r=DestindexCopyKv1.outputRegion
      · subst r
        exact DestindexCopyKv1.LogicalGrid.run_frame rest c ps s source dest h
          (fun p hp => (schedule p).mp hp) o
          (fun p hp head dim hh hd => ho p head dim ((schedule p).mp hp) hh hd)
      · exact DestindexCopyKv1.LogicalGrid.other_frame rest c ps s r o hr

#axiomsClean source_run
#axiomsClean run_memory
#axiomsClean complete
end
end VeriTile.Bench.Repairs.DestindexCopyKv2.LogicalGrid
