import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.SignedGrid

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.KernelRaw
open VeriTile Triton FP.Structural Direct FPValues
noncomputable section
variable {α : Type} [Inhabited α]

/-- A raw program's active scalar store; no Tensor metadata or allocation API. -/
def Outside (h : Call) (data : Data α) (r : RegionName) (off : Nat) : Prop :=
  r ≠ "Y" ∨ ∀ p : Geometry.Program h, data.indices p.batch ≠ -1 → off ≠ SourceGrid.address h p

theorem fold_reads (M : Algebra α) (R : Reduction α) (h : Call)
    (ps : List (Geometry.Program h)) (data : Data α) :
    (SourceGrid.fold M R h ps data).input = data.input ∧
    (SourceGrid.fold M R h ps data).weights = data.weights ∧
    (SourceGrid.fold M R h ps data).indices = data.indices ∧
    (SourceGrid.fold M R h ps data).other = data.other ∧
    (SourceGrid.fold M R h ps data).padding = data.padding := by
  induction ps generalizing data with
  | nil => exact ⟨rfl,rfl,rfl,rfl,rfl⟩
  | cons p rest ih => exact ih (SourceGrid.updated M R h p data)

theorem fold_frame (M : Algebra α) (R : Reduction α) (h : Call)
    (ps : List (Geometry.Program h)) (data : Data α) (off : Nat)
    (outside : ∀ p ∈ ps, data.indices p.batch ≠ -1 → off ≠ SourceGrid.address h p) :
    (SourceGrid.fold M R h ps data).output off = data.output off := by
  induction ps generalizing data with
  | nil => rfl
  | cons p rest ih =>
    rw [SourceGrid.fold,ih (SourceGrid.updated M R h p data)
      (fun q hq active => outside q (List.mem_cons_of_mem p hq) active)]
    by_cases active : data.indices p.batch ≠ -1
    · simp [SourceGrid.updated,active,outside p (by simp) active]
    · simp [SourceGrid.updated,active]

/-- Complete actual signed-source grid over raw arrays and geometry, with
all active values and the exact tagged memory frame, including skipped rows. -/
theorem complete (M : Algebra α) (R : Reduction α) (h : Call) (data : Data α)
    (valid : ∀ b, b < h.batches → data.indices b = -1 ∨ 0 ≤ data.indices b) :
    ∃ final, Signed.Grid.run M R h (Geometry.programs h) data = some final ∧
      (∀ p : Geometry.Program h, data.indices p.batch ≠ -1 →
        final.output (SourceGrid.address h p) = SourceGrid.value M R h p data) ∧
      (∀ r off, Outside h data r off → final.cells r off = data.cells r off) := by
  let final := SourceGrid.fold M R h (Geometry.programs h) data
  refine ⟨final,Signed.Grid.executes M R h _ data valid,?_,?_⟩
  · intro p active
    apply SourceGrid.values M R h _ _ data p (Geometry.programs_complete h p) active
    simpa [SourceGrid.address,Geometry.outputOffset,
      VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.Geometry.outputOffset] using Geometry.programs_unique h
  · intro r off outside
    have reads := fold_reads M R h (Geometry.programs h) data
    by_cases hy : r = "Y"
    · subst r
      have miss : ∀ p : Geometry.Program h, data.indices p.batch ≠ -1 → off ≠ SourceGrid.address h p :=
        outside.resolve_left (by simp)
      have keep := fold_frame M R h (Geometry.programs h) data off (fun p _ => miss p)
      simpa [Data.cells,final] using congrArg (fun x => Cell.mk TileDType.real x) keep
    · simp only [Data.cells,final,reads.1,reads.2.1,reads.2.2.1,reads.2.2.2.1]
      simp [hy]

#axiomsClean complete
#axiomsClean fold_reads
#axiomsClean fold_frame
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.KernelRaw
