import bench.repairs.tritonbench_g.add_example.IO
import bench.repairs.tritonbench_g.add_example.Kernels
import bench.repairs.tritonbench_g.add_example.Proofs.Contract
import bench.repairs.tritonbench_g.add_example.Proofs.Projection
import bench.tritonbench_g.add_example.Proofs.Real
import VeriTile.Triton.Memory.RegionalKernelIO
import VeriTile.Meta.StatementAudit


namespace VeriTile.Bench.Repairs.AddExample.Contiguous.ReaderProof
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
noncomputable section

theorem correct (n : Nat) :
    IO.addIO n ⊨ᵣ fun xs ys i => xs i + ys i :=by
  refine MaskedKernelIO₂.RegionalImplements.intro _ ?_ ?_
  · intro bounds s hx hy hout _
    exact VeriTile.Bench.TritonBenchG.AddExample.add_kernel_traceSafe
      "x" "y" "out" n blockSize bounds s hx hy hout
  · intro s xs ys hx hy
    obtain ⟨final, he, hv, hf⟩ := VeriTile.Bench.TritonBenchG.AddExample.add_kernel_region_run
      "x" "y" "out" n blockSize s xs ys hx hy
    exact ⟨final, he, hv, fun r i outside _ => hf r i outside⟩

#axiomsClean correct
end
end VeriTile.Bench.Repairs.AddExample.Contiguous.ReaderProof
