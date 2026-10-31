import bench.tritonbench_g.add_value.IO
import bench.tritonbench_g.add_value.Proofs.Mathematical_Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AddValue.Contiguous.ReaderProof
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

theorem correct (n : Nat) :
    IO.addValueIO n ⊨ fun xs i => xs i + 10 :=by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact VeriTile.Bench.TritonBenchG.AddValue.puzzle1_kernel_flattenOk
      "x" "out" n blockSize
  · intro bounds s hx hout _
    exact VeriTile.Bench.TritonBenchG.AddValue.puzzle1_kernel_traceSafe
      "x" "out" n blockSize bounds s hx hout
  · intro s xs hx
    obtain ⟨final, executed, values, frame⟩ :=
      VeriTile.Bench.TritonBenchG.AddValue.puzzle1_kernel_region_run
        "x" "out" n blockSize s xs hx
    exact ⟨final, executed, values, fun r i outside _ => frame r i outside⟩

#axiomsClean correct
end
end VeriTile.Bench.Repairs.AddValue.Contiguous.ReaderProof
