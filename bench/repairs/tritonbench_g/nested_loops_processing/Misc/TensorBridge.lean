import bench.repairs.tritonbench_g.nested_loops_processing.Correct
import bench.repairs.tritonbench_g.nested_loops_processing.FPEquiv
import bench.repairs.tritonbench_g.nested_loops_processing.Misc.KernelRealResult
import bench.repairs.tritonbench_g.nested_loops_processing.Misc.KernelNativeResult
import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.KernelResult

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.TensorBridge
open VeriTile.Triton FP.Structural
open KernelBinding KernelResult TypedExecution
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
noncomputable section

/-- Kernel-only checked entry: metadata and constructor words determine the
actual input state, and the complete source execution gives the independent
signed-int32 Real target while preserving every cell outside the output. -/
theorem nested_checked_kernel_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (api : API Word Store Byte) (memory : Store)
    (b : Binding api rows cols memory) (positive : 0<rows*cols) (context : State α) :
    ∃ final : State α,
      exec M (Kernels.source (rows*cols) cols) (b.state context) = some final ∧
      (∀ i, i<rows*cols → ((final.mem "output" i).read .int : ℝ) = realTarget rows cols i) ∧
      (∀ (r : RegionName) i, r ≠ "output" ∨ rows*cols ≤ i → final.mem r i = (b.state context).mem r i) :=
  KernelRealResult.kernel_correctness M rows cols positive (b.state context)
    (b.input_cell context) (b.output_cell context)

/-- Exact int32 equivalence, retaining the genuine commuted-multiplication
variant. Both actual full programs successfully return the same entire state. -/
theorem nested_checked_int32_kernel_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (api : API Word Store Byte) (memory : Store)
    (b : Binding api rows cols memory) (positive : 0<rows*cols) (context : State α) :
    ∃ final : State α,
      exec M (Kernels.source (rows*cols) cols) (b.state context) = some final ∧
      exec M (Kernels.commuted (rows*cols) cols) (b.state context) = some final ∧
      (∀ i, i<rows*cols → final.mem "output" i = .mk .int (signedValue (wordTarget rows cols i))) ∧
      (∀ i, i<rows*cols → BitVec.ofInt 32 ((final.mem "output" i).read .int) = wordTarget rows cols i) ∧
      (∀ (r : RegionName) i, r ≠ "output" ∨ rows*cols ≤ i → final.mem r i = (b.state context).mem r i) :=
  KernelNativeResult.kernel_equiv M rows cols positive (b.state context)
    (b.input_cell context) (b.output_cell context)

abbrev source_addresses := @Binding.pointer_bytes
abbrev physical_matches_typed := @typed_physical
abbrev physical_words_exact := @physical_words
abbrev physical_bytes_exact := @physical_bytes

#axiomsClean nested_checked_kernel_correctness
#axiomsClean nested_checked_int32_kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.TensorBridge
