import bench.repairs.tritonbench_g.matrix_vector_multip.Kernels
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeSource

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.KernelSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section

def parameters (N M BN BM : Nat) : Typed.Parameters :=
  ⟨"A", "B", "C", N, M, M, 1, 1, 1, BN, BM⟩

/-- Literal source binding, including retained FP32 precision tags. -/
theorem source_binding (N M BN BM : Nat) :
    Kernels.source N M BN BM = WholeSource.source (parameters N M BN BM) := rfl

def row (BN pid lane : Nat) : Nat := pid * BN + lane

def Outside (N BN pid : Nat) (r : RegionName) (o : Nat) : Prop :=
  r ≠ "C" ∨ ∀ i : Fin BN, row BN pid i.val < N → o ≠ row BN pid i.val

/-- Kernel-independent masked output scatter. This only defines the memory
effect of a supplied row formula; it does not execute the source. -/
def reference {α : Type} (N BN : Nat) (s : State α) (value : Fin BN → α) : State α :=
  (TileShape.allIndices [BN, 1]).foldl (fun t i =>
    if row BN (s.pids 0) i.1.val < N then
      t.write "C" (row BN (s.pids 0) i.1.val) (.mk .fp32 (value i.1))
    else t) s

theorem reference_value {α : Type} (N BN : Nat) (s : State α) (value : Fin BN → α)
    (i : Fin BN) (active : row BN (s.pids 0) i.val < N) :
    (reference N BN s value).mem "C" (row BN (s.pids 0) i.val) = .mk .fp32 (value i) := by
  apply State.masked_scatter_readback "C"
    (fun i : TileIndex [BN, 1] => row BN (s.pids 0) i.1.val)
    (fun i => Cell.mk .fp32 (value i.1))
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
      t.mem "C" (row BN (s.pids 0) i.val) = .mk .fp32 (value i))
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

/-- Complete source execution for arbitrary column extent, not a one-block
slice. The value graph retains every cast, masked zero and accumulator carry. -/
theorem execution {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM : Nat)
    (width : 0 < BM) (s : State α) :
    ∃ t, exec A (Kernels.source N M BN BM) s = some t ∧
      (∀ i : Fin BN, row BN (s.pids 0) i.val < N →
        t.mem "C" (row BN (s.pids 0) i.val) = .mk .fp32
          (WholeSource.tileValue A (parameters N M BN BM) (s.pids 0) s.mem i)) ∧
      (∀ r o, Outside N BN (s.pids 0) r o → t.mem r o = s.mem r o) := by
  rw [source_binding]
  exact WholeSource.tile_result A (parameters N M BN BM) width rfl s

#axiomsClean source_binding
#axiomsClean execution
#axiomsClean memory_ext
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.KernelSource
