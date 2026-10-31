import bench.tritonbench_g.matrix_vector_multip.Kernels
import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceReadback
import bench.tritonbench_g.matrix_vector_multip.Sources.Bounds

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.KernelSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section

def parameters (N M BN BM : Nat) : Typed.Parameters :=
  ⟨"A", "B", "C", N, M, M, 1, 1, 1, BN, BM⟩

/-- Literal source binding, including retained FP32 precision tags. -/
theorem source_binding (N M BN BM : Nat) :
    Kernels.source N M BN BM = Source.source (parameters N M BN BM) := rfl

def row (BN pid lane : Nat) : Nat := pid * BN + lane

def Outside (N BN pid : Nat) (r : RegionName) (o : Nat) : Prop :=
  r ≠ "C" ∨ ∀ i : Fin BN, row BN pid i.val < N → o ≠ row BN pid i.val

/-- Kernel-independent masked output scatter. This only defines the memory
effect of a supplied row formula; it does not execute the source. -/
def reference {α : Type} (N BN : Nat) (s : State α) (value : Fin BN → α) : State α :=
  (TileShape.allIndices [BN, 1]).foldl (fun t i =>
    if row BN (s.pids 0) i.1.val < N then
      t.write "C" (row BN (s.pids 0) i.1.val) (.mk .real (value i.1))
    else t) s

theorem reference_value {α : Type} (N BN : Nat) (s : State α) (value : Fin BN → α)
    (i : Fin BN) (active : row BN (s.pids 0) i.val < N) :
    (reference N BN s value).mem "C" (row BN (s.pids 0) i.val) = .mk .real (value i) := by
  apply State.masked_scatter_readback "C"
    (fun i : TileIndex [BN, 1] => row BN (s.pids 0) i.1.val)
    (fun i => Cell.mk .real (value i.1))
    (fun i => row BN (s.pids 0) i.1.val < N) s (i, 0, PUnit.unit) active
  simpa only [parameters, Nat.mul_one, WholeSource.row, row] using
    WholeSource.row_injective (parameters N 0 BN 1) (s.pids 0) rfl

theorem reference_frame {α : Type} (N BN : Nat) (s : State α) (value : Fin BN → α)
    (r : RegionName) (o : Nat) (outside : Outside N BN (s.pids 0) r o) :
    (reference N BN s value).mem r o = s.mem r o := by
  apply State.masked_scatter_frame
  rcases outside with hr | ho
  · exact Or.inl hr
  · exact Or.inr fun i _ hi => ho i.1 hi

theorem memory_ext {α : Type} (N BN : Nat) (s t : State α) (value : Fin BN → α)
    (values : ∀ i : Fin BN, row BN (s.pids 0) i.val < N →
      t.mem "C" (row BN (s.pids 0) i.val) = .mk .real (value i))
    (frame : ∀ r o, Outside N BN (s.pids 0) r o → t.mem r o = s.mem r o) :
    t.mem = (reference N BN s value).mem := by
  classical
  funext r o
  by_cases hit : ∃ i : Fin BN, r = "C" ∧ row BN (s.pids 0) i.val < N ∧ o = row BN (s.pids 0) i.val
  · obtain ⟨i, rfl, hi, rfl⟩ := hit
    rw [values i hi, reference_value N BN s value i hi]
  · have outside : Outside N BN (s.pids 0) r o := by
      by_cases hr : r = "C"
      · right
        intro i hi ho
        exact hit ⟨i, hr, hi, ho⟩
      · exact Or.inl hr
    rw [frame r o outside, reference_frame N BN s value r o outside]

theorem scalar_bounds (N M BN BM : Nat) (scope : SourceBounds N M BN BM) :
    ∀ name n, (name,n) ∈ Source.scalarParameters (parameters N M BN BM) → n<2^64 := by
  intro name n member
  simp only [Source.scalarParameters,parameters,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with h|h|h|h|h|h
  all_goals cases h
  · exact scope.scalar_rows
  · exact scope.scalar_columns
  · exact scope.scalar_columns
  all_goals norm_num

/-- Execute the complete repaired source, including its actual affine while,
all columns, readback and the exact exterior frame. -/
theorem execution {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM : Nat)
    (width : 0 < BM) (scope : SourceBounds N M BN BM)
    (castId : ∀ precision src dst value, A.cast precision src dst value=value)
    (loadId : ∀ value, A.fp32Load value=value) (s : State α) (pidSmall : s.pids 0<2^64) :
    ∃ t, exec A (Kernels.source N M BN BM) s = some t ∧
      (∀ i : Fin BN, row BN (s.pids 0) i.val < N →
        t.mem "C" (row BN (s.pids 0) i.val) = .mk .real
          (WholeSource.tileValue A (parameters N M BN BM) (s.pids 0) (Source.viewMemory s.mem) i)) ∧
      (∀ r o, Outside N BN (s.pids 0) r o → t.mem r o = s.mem r o) := by
  rw [source_binding]
  exact Source.source_correct A (parameters N M BN BM) width castId loadId s
    (scalar_bounds N M BN BM scope) pidSmall scope.lanes_rows scope.lanes_columns rfl
#axiomsClean source_binding
#axiomsClean execution
#axiomsClean memory_ext
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.KernelSource
