import bench.repairs.tritonbench_g.relu_strided_buffer.IO
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.ReaderBefore
namespace VeriTile.Bench.Repairs.ReluStridedBuffer.LogicalReader
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer SignedView Whole
noncomputable section
/-- Exact complete old/new judgment, not just agreement on a mathematical value. -/
theorem judgment_iff (rest : Algebra ℝ) (x y : SignedView.View) (n : Nat) :
    (LogicalView.reluIO rest x y n).Implements (fun xs i => max 0 (xs i)) ↔
    (ReluIO.kernelIO (realBackend rest) Bool.false x y n).Implements
      (ReluIO.pointwise (fun z => max 0 z) x y n) := Iff.rfl
end
end VeriTile.Bench.Repairs.ReluStridedBuffer.LogicalReader
