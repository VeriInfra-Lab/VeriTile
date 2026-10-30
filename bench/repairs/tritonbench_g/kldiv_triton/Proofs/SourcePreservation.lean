import bench.repairs.tritonbench_g.kldiv_triton.Kernels
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.LegacyPastedBody
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.KldivTriton.SourcePreservation
open VeriTile.Triton
/-- Entire source/candidate programs retain every old statement for all
runtime strides, dimensions, blocks, log modes and reductions. Their headers
now declare the actual input/output roles: the old collector lost those roles
when source pointer arguments were rebound before entering the loop. -/
theorem forward (Y T O : RegionName) (sy st so N B : Nat) (flag : Bool) (red : Nat) :
    forwardTemplate Y T O sy st so N B flag red =
      ComputeKernel.mk [Y,T] [O]
        (LegacyPastedBody.forwardTemplate Y T O sy st so N B flag red).surfaceBody := rfl

theorem backward (O T : RegionName) (so st N B : Nat) (flag : Bool) :
    backwardTemplate O T so st N B flag =
      ComputeKernel.mk [T] [O]
        (LegacyPastedBody.backwardTemplate O T so st N B flag).surfaceBody := rfl

theorem forwardCandidate (Y T O : RegionName) (sy st so N B : Nat) (flag : Bool) (red : Nat) :
    forwardCandidateTemplate Y T O sy st so N B flag red =
      ComputeKernel.mk [Y,T] [O]
        (LegacyPastedBody.forwardCandidateTemplate Y T O sy st so N B flag red).surfaceBody := rfl

theorem backwardCandidate (O T : RegionName) (so st N B : Nat) (flag : Bool) :
    backwardCandidateTemplate O T so st N B flag =
      ComputeKernel.mk [T] [O]
        (LegacyPastedBody.backwardCandidateTemplate O T so st N B flag).surfaceBody := rfl
#axiomsClean forward
#axiomsClean backward
#axiomsClean forwardCandidate
#axiomsClean backwardCandidate
end VeriTile.Bench.Repairs.KldivTriton.SourcePreservation
