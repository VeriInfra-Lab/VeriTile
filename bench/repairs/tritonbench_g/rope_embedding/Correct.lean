import bench.repairs.tritonbench_g.rope_embedding.Proofs.Interface
import bench.repairs.tritonbench_g.rope_embedding.Proofs.SourceBinding

namespace VeriTile.Bench.Repairs.RopeEmbedding
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Independent two-coordinate rotation. Coefficients need not lie on the
unit circle; the backward branch negates the sine coefficient. -/
def mathematical (r : Request) (back : Bool)
    (values : FastRopeEmbedding.LogicalIO.Inputs r) : Fin r.heads → Fin (2*r.half) → ℝ :=
  fun h col =>
    let i := FastRopeEmbedding.Interface.lane r col
    let a := values.query h (FastRopeEmbedding.Interface.left r i)
    let b := values.query h (FastRopeEmbedding.Interface.right r i)
    let c := values.cosine i
    let s := values.sine i
    let pair := if back then (a*c+b*s,b*c-a*s) else (a*c-b*s,b*c+a*s)
    if r.half ≤ col.val then pair.2 else pair.1

/-- Complete dynamic-loop execution, source access safety, both rotated halves
and the exact exterior cell frame, on the original launch domain. -/
specification rope_embedding_kernel_correct (r : Request) (back : Bool)
    (row group : Nat) (launch : r.Launched row group) :
    VeriTile.Spec.Real (ropeIO r back row group ⊨ mathematical r back) :=
  (Interface.judgment_iff r back row group _).mpr
    (Interface.real_implements r back row group launch)

#specNonCircular mathematical avoiding [Kernels.rope, Source.source, FP.Structural.exec]
#stmtSurfaceSubset rope_embedding_kernel_correct ⊆
  [VeriTile.Spec.Real, ropeIO, mathematical, LogicalProgramIO.Implements, Request, Request.Launched, FastRopeEmbedding.LogicalIO.Inputs,
   Request.heads,Request.half,Request.dimension]
#axiomsClean rope_embedding_kernel_correct
end
end VeriTile.Bench.Repairs.RopeEmbedding
