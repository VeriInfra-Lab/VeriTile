import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Misc.NativeIOResults
import bench.repairs.tritonbench_g.llama_ff_triton_scalar_cast.Proofs.RawNative

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.FPEquiv
open VeriTile Triton FP.Structural Execution
noncomputable section

def kernelIO (a : Words.Instructions) (p : Parameters) : ArrayProgramIO Words.F Words.F :=
  Interface.programIO Bool.true (Words.arithmetic a) (Words.gates a) p
abbrev formula := Reference.nativeFormula
def halfObservation (p : Parameters) :
    (o : Fin (Interface.outputPorts p).length) → TileIndex (Interface.outputPorts p)[o].shape →
      Words.F → Words.H := fun _ _ => Words.half

/-- Complete native half-word observation of the existing five compiler-bound
source stages. The original scalar/K16 ports and all int8/half payloads remain
admitted. This is the existing native result, with no floating rewrite claim. -/
theorem kernel_equiv (a : Words.Instructions) (p : Parameters) :
    (kernelIO a p).ObservesOn (halfObservation p) Interface.byteDomain (formula a p) :=
  Interface.Result.native a p

#stmtSurfaceSubset kernel_equiv ⊆
  [Words.Instructions,Words.F,Words.H,Parameters,ArrayProgramIO.ObservesOn,kernelIO,
   halfObservation,Interface.byteDomain,formula]
#specNonCircular formula avoiding [source,run,Grid.grid,Grid.one,nativeExecution,gateExecution]
#axiomsClean kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.FPEquiv
