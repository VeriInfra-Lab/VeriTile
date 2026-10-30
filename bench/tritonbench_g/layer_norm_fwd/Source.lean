import bench.tritonbench_g.layer_norm_fwd.Contract
import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedReturn

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section

def quoted (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ) : ComputeKernel :=
  Source.kernel (sourceFlags f) (sourceRegions r) (sourceStrides stride)
    M N eps p (tile N) (tile_arange N)

theorem source_exact (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ) :
    quoted f r stride M N eps p=repaired f r stride M N eps p (tile N) (tile_arange N) := rfl

/-- Complete shared source execution is proved for the repaired positive tile,
including N=0. No execution-success or arithmetic-result premise is supplied. -/
theorem source_row {α : Type} [Inhabited α] (A : Algebra α) (cmp : TypedSource.Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ) (s : State α)
    (sep : Domain.ReadWriteSeparated f r) (geometry : NativeGeometry.Geometry f r stride M N)
    (row : s.pids 0<M) :
    ∃ final, FP.Structural.exec (engine A) (quoted f r stride M N eps p) s=some final ∧
      TypedSource.RowPost A cmp f r stride M N eps p (tile N) (s.pids 0) s final :=
  TypedSource.source_row_post A cmp f r stride M N eps p (tile N) (tile_arange N) s sep geometry row

theorem zero_columns_writes (f : Flags) (r : Regions) (stride : Strides) (M row : Nat)
    (region : RegionName) (offset : Nat) :
    Contract.written f r stride M 0 row region offset ↔
      (region=r.rstd ∧ offset=row) ∨ (f.rms=Bool.false ∧ region=r.mean ∧ offset=row) := by
  have empty (start : Nat) : ¬(start≤offset ∧ offset<start) := by omega
  simp [Contract.written,Contract.inRow,empty]

#axiomsClean source_exact
#axiomsClean source_row
#axiomsClean zero_columns_writes
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
