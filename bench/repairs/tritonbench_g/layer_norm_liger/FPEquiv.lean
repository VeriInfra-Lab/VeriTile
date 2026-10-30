import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ProgramPair
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.LayerNormLiger.Kernel.FPEquiv
open VeriTile.Triton FP.Structural LayerIO
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive)
open scoped VeriTile.Spec
noncomputable section
variable {α : Type} [Inhabited α]

def forwardOriginal (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters) (pid : Nat) :=
  ProgramIO.Implementation.original (forwardIO M R p pid) (fun _ => True)
def forwardSharedCenter (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters) (pid : Nat) :=
  ProgramIO.Implementation.ofVariant (forwardCandidate M R p pid) (fun _ => True)
def backwardOriginal (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters) (pid : Nat) :=
  ProgramIO.Implementation.original (backwardIO M R p pid) (fun _ => True)
def backwardHoistedDenominator (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters) (pid : Nat) :=
  ProgramIO.Implementation.ofVariant (backwardCandidate M R p pid) (fun _ => True)

/-- The related candidate evaluates the identical FP32 centered vector once
instead of three times, preserving Y, mean, rstd, access safety and frame. -/
specification forward_pair (M : Algebra α) (R : SumPrimitive α) (p : Forward.Parameters) (pid : Nat) :
    forwardOriginal M R p pid ≡[[]] forwardSharedCenter M R p pid :=
  ProgramIO.Implementation.of_equivalent (forwardCandidate M R p pid) (ProgramPair.forward M R p pid)

/-- The related candidate moves the repeated scalar FP32 column-count conversion
outside the full row loop; DX, DW and DB retain every original operation/order. -/
specification backward_pair (M : Algebra α) (R : SumPrimitive α) (p : Backward.Parameters) (pid : Nat) :
    backwardOriginal M R p pid ≡[[]] backwardHoistedDenominator M R p pid :=
  ProgramIO.Implementation.of_equivalent (backwardCandidate M R p pid) (ProgramPair.backward M R p pid)

#axiomsClean forward_pair
#axiomsClean backward_pair
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormLiger.Kernel.FPEquiv
