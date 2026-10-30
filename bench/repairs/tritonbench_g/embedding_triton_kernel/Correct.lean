import bench.repairs.tritonbench_g.embedding_triton_kernel.IO
import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.LogicalIO
import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.HeadlineLookup

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The shared FP32 source, mixed metadata/payload channels, gather windows and
64-row launch are explicit in `IO.lean`. -/
def embeddingIO (rest : Algebra ℝ) (p : Parameters) (pid : Nat) :=
  LogicalIO.view rest p pid

/-- Complete source safety, successful execution, every active gather/zero
output and exterior typed-cell frame. Signed int32 metadata is the logical domain;
Real payloads are unrestricted. Layout requires separate output, covered columns and disjoint output rows. -/
specification embedding_kernel_correctness (rest : Algebra ℝ) (p : Parameters)
    (pid : Nat) (layout : Execution.Layout p) :
    Spec.Real ((embeddingIO rest p pid) ⊨{LogicalIO.Signed p pid}
      (fun xs row col => if p.first ≤ xs.ids row ∧ xs.ids row<p.last then
        xs.weightAt (xs.ids row-p.first).toNat col else 0)) :=
  (LogicalIO.judgment_iff rest p pid).mpr (IOImplements.implements_signed rest 0 Bool.false p pid layout)

#specNonCircular target avoiding [source,indexCommutedSource,FP.Structural.exec]
#axiomsClean embedding_kernel_correctness
#stmtSurfaceSubset embedding_kernel_correctness ⊆
  [LogicalIO.Inputs,Parameters.width,Spec.Real,embeddingIO,Execution.Layout,LogicalIO.Signed,LogicalIO.Inputs.ids,LogicalIO.Inputs.weightAt,
   Parameters,Parameters.first,Parameters.last,FP.Structural.Algebra,LogicalProgramIO.ImplementsOn]
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.Correct
