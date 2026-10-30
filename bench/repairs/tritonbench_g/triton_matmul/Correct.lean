import bench.repairs.tritonbench_g.triton_matmul.Proofs.CheckedMath
import bench.repairs.tritonbench_g.triton_matmul.Proofs.IdealHost
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Correct
open VeriTile.Triton FP.Structural
open Contract TensorBinding InputAdapter
noncomputable section

specification python_checks_iff_preconditions (a b : Torch.ArgumentMetadata) :
    checkInputs a b=.ok () ↔ AdmittedArguments a b := checkInputs_iff a b

/-- Arbitrary Real-valued memory: original tensor storage offsets and aliases,
checked metadata, fresh allocation, actual finite source/grid, returned C,
independent full-K matrix product, and the complete caller-memory frame. -/
specification matmul_checked_real (allocate : Allocator ℝ) (read : IdealTensor.RealRead)
    (a b : Tensor) (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (initial : State ℝ)
    (allocates : AllocationLaw allocate a b initial) (bound : IdealTensor.Inputs read a b initial) :
    VeriTile.Spec.Real (∃ c prepared final,allocate a b initial=some (c,prepared) ∧
      FiniteHost.call ideal idealDot allocate a b initial=some (c,final) ∧
      CheckedHost.Post a b c initial final (IdealTensor.expected read a b)) := by
  obtain ⟨c,prepared,final,allocated,called,post⟩ :=
    IdealHost.correct allocate read a b la lb h initial allocates bound
  refine ⟨c,prepared,final,allocated,called,post.metadata,?_,post.frame⟩
  intro i
  simpa only [IdealTensor.reference read a b la lb h initial bound i] using post.values i

/-- Independent mathematical GEMM under the explicit ideal Real interpretation.
The checked call executes finite scalar ABI, source controls, loads, full K
loop, grouped grid and stores. The returned C is the actual allocated object;
all exterior logical memory is preserved. -/
specification matmul_checked (allocate : Allocator ℝ) (read : Read) (decode : Word → ℝ)
    (a b : Tensor) (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (initial : State ℝ)
    (allocates : AllocationLaw allocate a b initial) (bound : Inputs read decode a b initial) :
    VeriTile.Spec.Real (∃ c prepared final,allocate a b initial=some (c,prepared) ∧
      FiniteHost.call ideal idealDot allocate a b initial=some (c,final) ∧
      CheckedHost.Post a b c initial final (CheckedMath.expected read decode a b la lb)) := by
  obtain ⟨c,prepared,final,allocated,called,post⟩ := FiniteHost.correct ideal idealDot idealNumerics
    allocate read decode a b la lb h (ideal_primitives _) initial allocates bound
  refine ⟨c,prepared,final,allocated,called,post.metadata,?_,post.frame⟩
  intro i
  simpa only [CheckedMath.reference read decode a b la lb h initial bound i] using post.values i

#axiomsClean python_checks_iff_preconditions
#axiomsClean matmul_checked_real
#axiomsClean matmul_checked
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Correct
