import bench.tritonbench_g.adam_update_triton.Kernels
import bench.examples.AdamUpdateGridLaunch.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Exact reuse of the example's mathematical grid theorem. This helper proves
one complete selected-config launch in the Real/Nat region model. It does not
model Python's CUDA assertions, autotuner trials/restore_value, scalar binding,
physical aliasing/strides, finite-width indexing, or a native runtime launch.
The existing FP contextual rewrite theorem is not an original-value theorem.
-/

namespace VeriTile.Bench.TritonBenchG.AdamUpdateTriton.GridReuse
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

/-- Explicit fp32 loads in the example and default loads in the benchmark
have exactly the same mathematical projection. No equality of their typed
floating executions, or identity law for a numerical load callback, is used. -/
theorem real_projection_eq (P G M : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) :
    (update_fn_kernel P G M lr wd beta1 beta2 n B).toAlgKernel =
      (VeriTile.Bench.Examples.AdamUpdateGridLaunch.Kernels.update_fn_kernel
        P G M lr wd beta1 beta2 n B).toAlgKernel := by
  rfl

/-- Both observable arrays belong to the same constructed final memory.
The independent formulas consume the initial memory. Both source autotune
block choices are covered, including an empty or partial final block. -/
theorem selected_config_real_grid (P G M : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat)
    (selected : B = 128 ∨ B = 1024) (distinct : P ≠ M) (initial : BlockState) :
    ∃ final : BlockState,
      Nonempty (Kernel.GridLaunchedOrdinary
        (update_fn_kernel P G M lr wd beta1 beta2 n B).toAlgKernel
        (VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamGrid ((n+B-1)/B)) initial final) ∧
      (∀ k, k < n →
        final.readMem P k =
          TiledOptimizer.lionParam (initial.readMem P k) (initial.readMem M k)
            (initial.readMem G k) lr wd beta1 ∧
        final.readMem M k =
          TiledOptimizer.lionMomentum (initial.readMem M k) (initial.readMem G k) beta2) := by
  have positive : 0 < B := by rcases selected with h | h <;> omega
  let count := (n+B-1)/B
  let final := Kernel.mergeFrames
    (VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamGrid count) initial
    (fun idx => VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamFrame
      P G M lr wd beta1 beta2 n B count distinct initial idx)
  refine ⟨final, ?_, ?_⟩
  · rw [real_projection_eq]
    exact ⟨VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamLaunch
      P G M lr wd beta1 beta2 n B ((n+B-1)/B) positive distinct initial⟩
  · exact VeriTile.Bench.Examples.AdamUpdateGridLaunch.update_fn_kernel_grid_correct_unconditional
      P G M lr wd beta1 beta2 n B positive distinct initial

#axiomsClean real_projection_eq
#axiomsClean selected_config_real_grid
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.AdamUpdateTriton.GridReuse
