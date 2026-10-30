import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.IO
import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.CandidateKernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

/-! The shifted candidate changes only the source field of the shared invocation. -/
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.Pair
open VeriTile Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def invocation (g : Geometry) (M : Algebra α) (D : DotAcc α) (pid : Nat) : Invocation α :=
  { PublicIO.invocation g M D pid with kernel := CandidateSource.typed g.request.parameters g.storage }

def io (rest : Algebra Word) (I : Native.Instructions) (g : Geometry) (ids : List Nat) :=
  PublicIO.io g (Native.algebra rest I) (Native.dotAcc I) ids

def variant (rest : Algebra Word) (I : Native.Instructions) (g : Geometry) (ids : List Nat) :
    ProgramIO.Variant (io rest I g ids) where
  schedule := fun _ => ids.map (invocation g (Native.algebra rest I) (Native.dotAcc I))
  observe := fun _ _ => id

end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.Pair
