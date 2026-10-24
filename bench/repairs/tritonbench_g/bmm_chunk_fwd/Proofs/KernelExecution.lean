import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.KernelExecutionRaw

namespace VeriTile.Bench.Repairs.BmmChunkFwd.KernelExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open DirectFP NativeGraph
noncomputable section

def source (rest : Algebra Expr) (p : Metadata) (chosen : Host.Config)
    (s : State Expr) : Option (State Expr) := Raw.source rest (call p) chosen s

def candidate (rest : Algebra Expr) (p : Metadata) (chosen : Host.Config)
    (s : State Expr) : Option (State Expr) := Raw.candidate rest (call p) chosen s

theorem equivalent (rest : Algebra Expr) (p : Metadata) (chosen : Host.Config) (s : State Expr) :
    source rest p chosen s=candidate rest p chosen s :=
  Raw.equivalent rest (call p) chosen rfl rfl s

theorem complete (rest : Algebra Expr) (p : Metadata) (chosen : Host.Config)
    (s : State Expr) (hm : Memory s) :
    ∃ t,source rest p chosen s=some t ∧ candidate rest p chosen s=some t ∧
      t.mem=NativeReference.completeMemory p chosen s.mem ∧
      Memory t ∧ FPSuccess.RegionFrame (call p).Out s t :=
  Raw.complete rest (call p) chosen rfl rfl s hm

#axiomsClean equivalent
#axiomsClean complete
end
end VeriTile.Bench.Repairs.BmmChunkFwd.KernelExecution
