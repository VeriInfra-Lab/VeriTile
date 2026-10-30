import bench.repairs.tritonbench_g.l2_norm_triton2.IO
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.GridIO
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.GridIO

namespace VeriTile.Bench.Repairs.L2NormTriton2.Pair
open VeriTile.Triton FP.Structural FloatingIO
open scoped VeriTile.Spec
noncomputable section

theorem forward {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) :
    forwardOriginal A rows N B eps ≡[[]] forwardCandidate A rows N B eps :=
  ProgramIO.Implementation.of_equivalent (GridIO.forwardCandidate A rows N B eps)
    (L2NormTriton1.GridIO.equivalence A rows N B eps)

theorem backward {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) :
    backwardOriginal A rows N B eps ≡[[]] backwardCandidate A rows N B eps :=
  ProgramIO.Implementation.of_equivalent (GridIO.candidate A rows N B eps)
    (GridIO.equivalence A rows N B eps)

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.L2NormTriton2.Pair
