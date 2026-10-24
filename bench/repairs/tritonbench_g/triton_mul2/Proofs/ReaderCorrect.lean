import bench.repairs.tritonbench_g.triton_mul2.IO
import bench.repairs.tritonbench_g.triton_mul2.Kernels
import bench.repairs.tritonbench_g.triton_mul2.Proofs.Projection
import bench.repairs.tritonbench_g.triton_mul2.Contract
import bench.tritonbench_g.triton_mul2.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! The ordinary and in-place source kernels implement the real formula 2 * xs i.
The two commuted candidate results are retained in Misc/CandidateCorrect.lean.
The masked IO contracts include termination, active-address bounds and the
memory frame. The in-place contracts read the original values before writing
back to the same region. The host grid composition is outside these contracts. -/

namespace VeriTile.Bench.Repairs.TritonMul2.ReaderCorrect

open VeriTile.Triton VeriTile.Examples
open VeriTile.Bench.TritonBenchG.TritonMul2
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Symbolic windows and tail mask for ordinary output. -/
def mul2IO (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ :=
  IO.mul2IO in_ptr0 out_ptr n_elements BLOCK_SIZE

theorem mul2_kernel_correctness
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    Spec.Real (mul2IO in_ptr0 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact mul2_kernel_flattenOk in_ptr0 out_ptr n_elements BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact mul2_kernel_traceSafe in_ptr0 out_ptr n_elements BLOCK_SIZE
      bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := mul2_kernel_region_run
      in_ptr0 out_ptr n_elements BLOCK_SIZE s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- Both roles share the same region for in-place output. -/
def mul2InplaceIO (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ :=
  IO.mul2InplaceIO ptr n_elements BLOCK_SIZE

theorem mul2_inplace_kernel_correctness
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    Spec.Real (mul2InplaceIO ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact mul2_inplace_kernel_flattenOk ptr n_elements BLOCK_SIZE
  · intro bounds s h1 _ _
    exact mul2_inplace_kernel_traceSafe ptr n_elements BLOCK_SIZE
      bounds s h1
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := mul2_inplace_kernel_region_run
      ptr n_elements BLOCK_SIZE s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

#stmtSurfaceSubset mul2_kernel_correctness ⊆
  [Spec.Real, RegionName, mul2IO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B]
#stmtSurfaceSubset mul2_inplace_kernel_correctness ⊆
  [Spec.Real, RegionName, mul2InplaceIO, MaskedKernelIO₁.Implements, MaskedKernelIO₁.B]

#specNonCircular mathematical avoiding [sourceKernel, commutedSourceKernel]
#axiomsClean mul2_kernel_correctness
#axiomsClean mul2_inplace_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.TritonMul2.ReaderCorrect
