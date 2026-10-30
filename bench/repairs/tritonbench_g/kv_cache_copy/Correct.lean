import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.KvCacheCopy.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- The complete source copies both logical K/V tiles through the declared
metadata-dependent windows, under the existing output-injectivity condition. -/
specification kv_cache_copy_real (rest : Algebra ℝ) (p : Source.Parameters)
    (seqPid headPid : Nat) :
    Spec.Real (LogicalIO.io rest p seqPid headPid ⊨{
      fun a => CopyIO.OutputUnique p headPid a.length a.block}
      fun a => ⟨a.key,a.value⟩) :=
  LogicalIO.implements rest p seqPid headPid

#axiomsClean kv_cache_copy_real
#specNonCircular CopyIO.outputAddress avoiding [Source.originalKernel, Source.kernel, FP.Structural.exec]
#specNonCircular CopyIO.copyValues avoiding [Source.originalKernel, Source.kernel, FP.Structural.exec]
end
end VeriTile.Bench.Repairs.KvCacheCopy.Correct
