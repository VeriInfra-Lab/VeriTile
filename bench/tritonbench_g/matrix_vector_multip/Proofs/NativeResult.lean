import bench.tritonbench_g.matrix_vector_multip.Proofs.IO
import bench.tritonbench_g.matrix_vector_multip.Proofs.KernelFP

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.KernelNativeResult
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section

def precision : FloatDType := .fp32

/-- Actual intended FP equivalence for this task: the typed full-loop source
versus the independent ordered native FMA/add graph. Primitive semantics and
compiler graph binding are the model boundary, not deferred rewrite laws.
There is no claim that different reduction trees or split mul/add agree. -/
theorem mv_fp32_kernel_equiv (native : Scalars) (N M BN BM : Nat)
    (g : Graph BN BM) (width : 0 < BM) (s : State Payload)
    (matrix : Nat → Nat → BitVec 32) (vector : Nat → BitVec 32)
    (inputs : KernelFP.InputWords N M s matrix vector) :
    ∃ t, exec (algebra native g) (Kernels.source N M BN BM) s = some t ∧
      t.mem = (reference N BN s (fun i => Payload.word
        (KernelFP.formula native N M BN BM g matrix vector (s.pids 0) i))).mem ∧
      (∀ i : Fin BN, row BN (s.pids 0) i.val < N →
        t.mem "C" (row BN (s.pids 0) i.val) = .mk .fp32 (.word
          (KernelFP.formula native N M BN BM g matrix vector (s.pids 0) i))) ∧
      (∀ r o, Outside N BN (s.pids 0) r o → t.mem r o = s.mem r o) := by
  obtain ⟨t, ran, values, frame⟩ := execution (algebra native g) N M BN BM width s
  have results : ∀ i : Fin BN, row BN (s.pids 0) i.val < N →
      t.mem "C" (row BN (s.pids 0) i.val) = .mk .fp32 (.word
        (KernelFP.formula native N M BN BM g matrix vector (s.pids 0) i)) := by
    intro i hi
    rw [values i hi, KernelFP.formula_eq native g inputs]
  exact ⟨t, ran, memory_ext N BN s t _ results frame, results, frame⟩

/-- Same authoritative source in both public entries; retained FP32 casts,
initial accumulator and reduction are never erased for this equivalence. -/
theorem source_binding (N M BN BM : Nat) :
    Kernels.source N M BN BM = WholeSource.source (parameters N M BN BM) :=
  KernelSource.source_binding N M BN BM

theorem io_correct (native : Scalars) (N M BN BM pid : Nat)
    (g : Graph BN BM) (width : 0 < BM) :
    (IO.io (algebra native g) N M BN BM pid Payload.word Payload.bits).Implements
      (fun vals => IO.outputValues (KernelFP.formula native N M BN BM g
        (IO.matrixValues vals) (IO.vectorValues vals) pid)) := by
  apply IO.implements (algebra native g) N M BN BM pid width Payload.word Payload.bits
    (fun matrix vector => KernelFP.formula native N M BN BM g matrix vector pid)
  intro s matrix vector matrixRead vectorRead hpid
  obtain ⟨t, ran, _, values, frame⟩ :=
    mv_fp32_kernel_equiv native N M BN BM g width s matrix vector
      ⟨matrixRead, vectorRead⟩
  refine ⟨t, ran, ?_, ?_⟩
  · intro i active
    have result := values i (by simpa [row, hpid] using active)
    simp only [row, hpid] at result
    rw [result]
    rfl
  · intro r o outside
    exact frame r o (by simpa [hpid] using outside)

#axiomsClean io_correct
#specNonCircular KernelFP.formula avoiding [Kernels.source, FP.Structural.exec]
#axiomsClean mv_fp32_kernel_equiv
#axiomsClean source_binding
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.KernelNativeResult
