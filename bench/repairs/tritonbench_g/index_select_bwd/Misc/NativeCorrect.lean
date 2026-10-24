import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Interface
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.IndexSelectBwd.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Torch.FP32 (Word)
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section
local instance wordInhabited : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The same FP32 source and IO windows, interpreted by the selected native
addition primitive. Every FP32 word is admitted, including exceptional values. -/
def indexIO (rest : Algebra Word) (nativeAdd : Word → Word → Word)
    (p : Launch) : ProgramIO Word Word := Interface.programIO rest nativeAdd p

/-- Independent native operation target; no reassociation or source
transformation is introduced. The primitive interpretation is a model boundary. -/
def native (nativeAdd : Word → Word → Word) (p : Launch)
    (values : Interface.Inputs Word p) : Interface.Outputs Word p :=
  Interface.result (fun i => nativeAdd (Interface.current values i) (Interface.incoming values i))

theorem index_select_bwd_fp32_equiv (rest : Algebra Word)
    (nativeAdd : Word → Word → Word) (p : Launch)
    (stride : p.stride = 1) (count : p.count ≤ p.block) :
    indexIO rest nativeAdd p ⊨ native nativeAdd p :=
  Interface.implements rest nativeAdd p stride count

#specNonCircular native avoiding [kernel, Source.execution, FP.Structural.exec]
#axiomsClean index_select_bwd_fp32_equiv
#stmtSurfaceSubset index_select_bwd_fp32_equiv ⊆
  [indexIO, native, ProgramIO.Implements, FP.Structural.Algebra, Word, wordInhabited,
    Launch, Launch.stride, Launch.count, Launch.block]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IndexSelectBwd.FPEquiv
