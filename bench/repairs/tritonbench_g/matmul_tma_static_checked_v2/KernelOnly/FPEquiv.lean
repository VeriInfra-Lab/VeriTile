import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Proofs.IOPair

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
open VeriTile.Triton FP.Structural FPValue Host StaticChecked.Native
open scoped VeriTile.Spec
noncomputable section
namespace FPEquiv
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def original (rest : Algebra Word) (I : Instructions) (profile : Profile) (h : Shape) (f : Flags) :=
  ProgramIO.Implementation.original (IOPair.io rest I profile h f) (fun _ => True)
def candidate (rest : Algebra Word) (I : Instructions) (profile : Profile) (h : Shape) (f : Flags) :=
  ProgramIO.Implementation.ofVariant (IOPair.variant rest I profile h f) (fun _ => True)

/-- Compare the complete original body with its historical redundant FP32
output-conversion variant. The original has no FP32 cast to remove. FP16
conversion and all fixed-K16 dot instructions remain, for both transpose flags
and all input words. Both programs are safe, succeed, agree on active typed
outputs and preserve both exterior frames. The existing local native cast
identity proves the same conversion relation in the corrected orientation. -/
specification kernel_equivalent (rest : Algebra Word) (I : Instructions)
    (profile : Profile) (h : Shape) (f : Flags) :
    original rest I profile h f ≡[[]] candidate rest I profile h f := by
  apply ProgramIO.Implementation.of_equivalent
  exact IOPair.equivalent rest I profile h f

#axiomsClean kernel_equivalent
end FPEquiv
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
