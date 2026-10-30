import bench.tritonbench_g.matmul_tma.Proofs.IOPair

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

/-- Remove the redundant same-format output cast in the FP32 branch of the
complete source. FP16 conversion and every fixed-K16 dot instruction are
retained, for both transpose flags and all input words. Both programs are safe,
succeed, agree on every active typed output and preserve both exterior frames. -/
specification kernel_equivalent (rest : Algebra Word) (I : Instructions)
    (profile : Profile) (h : Shape) (f : Flags) :
    original rest I profile h f ≡[[]] candidate rest I profile h f := by
  apply ProgramIO.Implementation.of_equivalent
  exact IOPair.equivalent rest I profile h f

#axiomsClean kernel_equivalent
end FPEquiv
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
