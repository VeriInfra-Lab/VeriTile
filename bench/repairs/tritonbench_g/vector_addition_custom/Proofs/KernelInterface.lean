import bench.repairs.tritonbench_g.vector_addition_custom.IO
import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.Contract
import bench.tritonbench_g.vector_addition_custom.Proofs.Real

namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.KernelInterface
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
noncomputable section

/-- Canonical-source Real correctness: successful execution, the independent
formula at every active lane, and the frame outside active output addresses.
Entry reads are required only where the actual source mask is true. The two
read-only regions may be observations of overlapping Tensor storage. -/
theorem region_correctness (n : Nat) (s : BlockState)
    (xs ys : Nat → ℝ)
    (hx : ∀ j : Fin blockSize, s.pid * blockSize + j.val < n →
      s.readMem "x" (s.pid * blockSize + j.val) = xs (s.pid * blockSize + j.val))
    (hy : ∀ j : Fin blockSize, s.pid * blockSize + j.val < n →
      s.readMem "y" (s.pid * blockSize + j.val) = ys (s.pid * blockSize + j.val)) :
    Spec.Real (∃ final, exec (originalKernel n).toAlgKernel s = some final ∧
      (∀ j : Fin blockSize, s.pid * blockSize + j.val < n →
        final.readMem "out" (s.pid * blockSize + j.val) =
          formula xs ys (s.pid * blockSize + j.val)) ∧
      (∀ (r : RegionName) (o : Nat), (r ≠ "out" ∨ ∀ j : Fin blockSize,
        s.pid * blockSize + j.val < n → o ≠ s.pid * blockSize + j.val) →
        final.mem r o = s.mem r o)) := by
  exact VeriTile.Bench.TritonBenchG.VectorAdditionCustom.add_kernel_region_run
    "x" "y" "out" n blockSize s
    (fun j => xs (s.pid * blockSize + j.val))
    (fun j => ys (s.pid * blockSize + j.val)) hx hy

/-- Active accesses are bounded independently of separation between regions. -/
theorem access_safe (n : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : n ≤ bounds "x") (hy : n ≤ bounds "y") (hout : n ≤ bounds "out") :
    Kernel.TraceSafe bounds (originalKernel n).toAlgKernel s :=
  VeriTile.Bench.TritonBenchG.VectorAdditionCustom.add_kernel_traceSafe
    "x" "y" "out" n blockSize bounds s
    (fun _ active => Nat.lt_of_lt_of_le active hx)
    (fun _ active => Nat.lt_of_lt_of_le active hy)
    (fun _ active => Nat.lt_of_lt_of_le active hout)

/-- Optional flat-memory corollary. `MaskedKernelIO₂.Implements` requires
pairwise disjoint allocations for x, y and out; it is not the alias-permitting
Python contract. The primary regional IO has no such requirement. -/
theorem flat_correctness (n : Nat) :
    Spec.Real (addIO n ⊨ fun xs ys i => xs i + ys i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact VeriTile.Bench.TritonBenchG.VectorAdditionCustom.add_kernel_flattenOk
      "x" "y" "out" n blockSize
  · intro bounds s hx hy hout _
    exact VeriTile.Bench.TritonBenchG.VectorAdditionCustom.add_kernel_traceSafe
      "x" "y" "out" n blockSize bounds s hx hy hout
  · intro s xs ys hx hy
    obtain ⟨final, he, hv, hf⟩ := VeriTile.Bench.TritonBenchG.VectorAdditionCustom.add_kernel_region_run
      "x" "y" "out" n blockSize s xs ys hx hy
    exact ⟨final, he, hv, fun r i outside _ => hf r i outside⟩

#axiomsClean flat_correctness
#axiomsClean region_correctness
#axiomsClean access_safe
end
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.KernelInterface
