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

/-- Inline the original optional FP16 conversion into the selected store.
The FP32 branch has no explicit source conversion to remove. Every fixed-K16
dot instruction, both transpose flags, all input words, active typed outputs,
safe successful executions and both exterior memory frames are retained. -/
specification kernel_equivalent (rest : Algebra Word) (I : Instructions)
    (profile : Profile) (h : Shape) (f : Flags) :
    original rest I profile h f ≡[[]] candidate rest I profile h f := by
  apply ProgramIO.Implementation.of_equivalent
  exact IOPair.equivalent rest I profile h f

#print_fp_assumptions kernel_equivalent
#axiomsClean kernel_equivalent
end FPEquiv
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
