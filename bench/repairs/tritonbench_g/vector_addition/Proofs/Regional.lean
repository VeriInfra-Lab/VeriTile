import bench.repairs.tritonbench_g.vector_addition.Kernels
import bench.repairs.tritonbench_g.vector_addition.Proofs.Contract
import bench.repairs.tritonbench_g.vector_addition.Proofs.Projection
import bench.tritonbench_g.vector_addition.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous.Regional
open VeriTile.Triton
noncomputable section

/-- Canonical-source Real correctness: successful execution, the independent
formula at every active lane, and the frame outside active output addresses.
Entry reads are required only where the actual source mask is true. The two
read-only regions may be observations of overlapping Tensor storage. -/
theorem add_kernel_correctness (n : Nat) (s : BlockState)
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
  exact VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel_region_run
    "x" "y" "out" n blockSize s
    (fun j => xs (s.pid * blockSize + j.val))
    (fun j => ys (s.pid * blockSize + j.val)) hx hy

/-- Active accesses are bounded independently of separation between regions. -/
theorem add_kernel_access_safe (n : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : n ≤ bounds "x") (hy : n ≤ bounds "y") (hout : n ≤ bounds "out") :
    Kernel.TraceSafe bounds (originalKernel n).toAlgKernel s :=
  VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel_traceSafe
    "x" "y" "out" n blockSize bounds s
    (fun _ active => Nat.lt_of_lt_of_le active hx)
    (fun _ active => Nat.lt_of_lt_of_le active hy)
    (fun _ active => Nat.lt_of_lt_of_le active hout)

#axiomsClean add_kernel_correctness
#axiomsClean add_kernel_access_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAddition.Contiguous.Regional
