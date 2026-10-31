import bench.tritonbench_g.square_matrix.Proofs.Contract
import bench.tritonbench_g.square_matrix.IO
import bench.tritonbench_g.square_matrix.Proofs.Mathematical_Real
import bench.tritonbench_g.square_matrix.Proofs.Mathematical_Source

namespace VeriTile.Bench.Repairs.SquareMatrix.KernelReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SquareMatrix
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

theorem source_projection (n B : Nat) :
    (fp32Kernel n B).eraseDType.toAlgKernel.body =
      (VeriTile.Bench.TritonBenchG.SquareMatrix.square_kernel "out" "x" n n n B).toAlgKernel.body := by
  change (typedKernel Bool.false "out" "x" n n n B).eraseDType.toAlgKernel.body = _
  rw [typed_projection]
  simp only [mathKernel, VeriTile.Bench.TritonBenchG.SquareMatrix.square_kernel,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]
  all_goals rfl

/-- One program owns one contiguous row, with padding excluded by both masks.
This IO uses the same canonical FP32 source as the equivalence proof. -/
def squareIO (n B : Nat) : MaskedKernelIO₁ := IO.squareIO n B

/-- Successful execution, active row outputs, access safety and exterior frame.
No host runner or allocation postcondition supplies any output value. -/
theorem implements (n B : Nat) :
    squareIO n B ⊨ fun xs i => mathematical (xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · change ((fp32Kernel n B).eraseDType.toAlgKernel).FlattenOk
    simpa only [Kernel.FlattenOk, source_projection] using
      square_kernel_flattenOk "out" "x" n n n B
  · intro bounds s hi ho _
    change Kernel.TraceSafe bounds ((fp32Kernel n B).eraseDType.toAlgKernel) s
    simpa only [Kernel.TraceSafe, source_projection] using
      square_kernel_traceSafe "out" "x" n n n B bounds s hi ho
  · intro s xs hx
    obtain ⟨t, he, hv, hf⟩ := square_kernel_region_run "out" "x" n n n B s xs hx
    refine ⟨t, ?_, hv, fun r o outside _ => hf r o outside⟩
    change exec ((fp32Kernel n B).eraseDType.toAlgKernel) s = some t
    simpa only [exec, source_projection] using he

/-- The row-wide formula on initial tensor coordinates, including every
logical column because the wrapper-selected block covers the whole row. -/
theorem coordinates (n : Nat) (s : BlockState) :
    ∃ t, exec ((fp32Kernel n (block n)).eraseDType.toAlgKernel) s = some t ∧
      (∀ col, col < n → t.readMem "out" (s.pids 0*n+col) =
        mathematical (s.readMem "x" (s.pids 0*n+col))) ∧
      (∀ (r : RegionName) o,
        r ≠ "out" ∨ o < s.pids 0*n ∨ s.pids 0*n+n ≤ o → t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := square_kernel_region_run "out" "x" n n n (block n) s
    (fun i => s.readMem "x" (s.pids 0*n+i.val)) (by intro i _; rfl)
  refine ⟨t, ?_, ?_, ?_⟩
  · simpa only [exec, source_projection] using he
  · intro col active
    exact hv ⟨col, active.trans_le (block_covers n)⟩ active
  · intro r o outside
    apply hf r o
    rcases outside with region | lower | upper
    · exact Or.inl region
    · exact Or.inr (fun i _ eq => by change o = s.pids 0*n+i.val at eq; omega)
    · exact Or.inr (fun i hi eq => by change o = s.pids 0*n+i.val at eq; omega)

#axiomsClean source_projection
#axiomsClean implements
#axiomsClean coordinates
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SquareMatrix.KernelReal
