import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.IO
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelSource
import bench.tritonbench_g.matrix_vector_multip.Proofs.WholeMath

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.KernelRealResult
open VeriTile VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section

/-- Independent full row dot product, with no block size or source executor. -/
def matrixVector (M : Nat) (matrix : Nat → Nat → ℝ) (vector : Nat → ℝ) (i : Nat) : ℝ :=
  ∑ j : Fin M, matrix i j.val * vector j.val


/-- The same precision-tagged source is interpreted over Real values. Every
column-loop iteration, zero extent and tail is included; the conclusion gives
successful execution, complete typed memory, active outputs and the frame. -/
theorem mv_kernel_correctness (N M BN BM : Nat) (width : 0 < BM)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (s : State ℝ) (domain : SourceDomain N M BN BM (s.pids 0)) (matrix : Nat → Nat → ℝ) (vector : Nat → ℝ)
    (matrixRead : ∀ i j, i < N → j < M →
      (s.mem "A" (i * M + j)).read .fp32 = matrix i j)
    (vectorRead : ∀ j, j < M → (s.mem "B" j).read .fp32 = vector j) :
    Spec.Real (∃ t, exec WholeMath.algebra (Kernels.source N M BN BM rowShape columnShape) s = some t ∧
      t.mem = (reference N BN s (fun i => matrixVector M matrix vector (row BN (s.pids 0) i.val))).mem ∧
      (∀ i : Fin BN, row BN (s.pids 0) i.val < N →
        t.mem "C" (row BN (s.pids 0) i.val) = .mk .fp32
          (matrixVector M matrix vector (row BN (s.pids 0) i.val))) ∧
      (∀ r o, Outside N BN (s.pids 0) r o → t.mem r o = s.mem r o)) := by
  obtain ⟨t, ran, values, frame⟩ := execution WholeMath.algebra N M BN BM width rowShape columnShape s domain
  have mathematical : ∀ i : Fin BN, row BN (s.pids 0) i.val < N →
      WholeSource.tileValue WholeMath.algebra (parameters N M BN BM) (s.pids 0) s.mem i =
        matrixVector M matrix vector (row BN (s.pids 0) i.val) := by
    intro i active
    rw [WholeMath.tile_mathematical (parameters N M BN BM) width _ _ i active]
    unfold matrixVector
    apply Finset.sum_congr rfl
    intro j _
    change (s.mem "A" (row BN (s.pids 0) i.val * M + j.val * 1)).read .fp32 *
      (s.mem "B" (j.val * 1)).read .fp32 = _
    simp only [Nat.mul_one]
    rw [matrixRead _ _ active j.isLt, vectorRead _ j.isLt]
  have results : ∀ i : Fin BN, row BN (s.pids 0) i.val < N →
      t.mem "C" (row BN (s.pids 0) i.val) = .mk .fp32
        (matrixVector M matrix vector (row BN (s.pids 0) i.val)) := by
    intro i hi
    rw [values i hi, mathematical i hi]
  exact ⟨t, ran, memory_ext N BN s t _ results frame, results, frame⟩

theorem io_correct (N M BN BM pid : Nat) (width : 0 < BM)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) (domain : SourceDomain N M BN BM pid) :
    (IO.io WholeMath.algebra N M BN BM pid rowShape columnShape (id : ℝ → ℝ) id).Implements
      (fun vals => IO.outputValues (fun i => matrixVector M
        (IO.matrixValues vals) (IO.vectorValues vals) (pid * BN + i.val))) := by
  apply IO.implements WholeMath.algebra N M BN BM pid rowShape columnShape width domain id id
    (fun matrix vector i => matrixVector M matrix vector (pid * BN + i.val))
  intro s matrix vector matrixRead vectorRead hpid
  obtain ⟨t, ran, _, values, frame⟩ :=
    mv_kernel_correctness N M BN BM width rowShape columnShape s (by simpa [hpid] using domain) matrix vector matrixRead vectorRead
  refine ⟨t, ran, ?_, ?_⟩
  · intro i active
    have result := values i (by simpa [row, hpid] using active)
    simp only [row, hpid] at result
    rw [result]
    rfl
  · intro r o outside
    exact frame r o (by simpa [hpid] using outside)

#axiomsClean io_correct
#specNonCircular matrixVector avoiding [Kernels.source]
#axiomsClean mv_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.KernelRealResult
