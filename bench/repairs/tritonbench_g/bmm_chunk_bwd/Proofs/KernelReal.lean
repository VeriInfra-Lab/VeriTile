import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Real

namespace VeriTile.Bench.Repairs.BmmChunkBwd.KernelReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section
open Classical

/-- Finite kernel-grid execution on already bound memory. Empty tensors have
no programs. Allocation, autotuning and the Python wrapper are not executed. -/
def run (p : Metadata) (a : Direct.Allocation p) (choice : Fin 9) (initial : BlockState) :
    Option BlockState :=
  if (p.tensor .a).numel=0 then some initial else
    HostGrid.run ((Direct.flat p a).flattenKernel
      (Kernels.fp32Source (call p) (tuning choice)).eraseDType.toAlgKernel)
      (HostContract.grid (call p) (tuning choice))
      (HostGrid.programs (HostContract.grid (call p) (tuning choice))) initial

/-- Existing complete Real proof reused with exactly one kernel grid. -/
theorem complete (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (inputs : Mathematical.Inputs) (choice : Fin 9) :
    ∃ final,run p a choice ((Direct.flat p a).flattenState (Direct.realState p inputs))=some final ∧
      (∀ b s g n,b<batch p → s<seqlen p → g<groups p → n<width p →
        final.readMem (Direct.flat p a).flat
          ((Direct.flat p a).addr (call p).Out (Direct.tensorIndex p b s g n))=
            Mathematical.product p inputs b s g n) ∧
      (∀ r off,¬Direct.OutputCell p a r off →
        final.mem r off=((Direct.flat p a).flattenState (Direct.realState p inputs)).mem r off) := by
  obtain ⟨final,success,values,frame⟩ := Direct.real_complete p h a inputs [] choice
  refine ⟨final,?_,values,frame⟩
  simpa only [run,Kernels.fp32Source_eq (call p) (tuning choice) rfl rfl,
    Direct.executeReal,RealTrials.flatHistory,Option.bind_some,RealTrials.flatOne] using success

#axiomsClean complete
#specNonCircular Mathematical.product avoiding [run,Kernels.fp32Source]
end
end VeriTile.Bench.Repairs.BmmChunkBwd.KernelReal
