import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.NativeReferenceProgram
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.FP

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeReference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open NativeGraph DirectFP
noncomputable section

def clearMemory (h : Host.Call) (memory : RegionName → Nat → Cell Expr) : RegionName → Nat → Cell Expr :=
  if h.causal then fun r off => if r=h.Out then
    Cleanup.run h (.mk .fp32 (ofFloat .fp32 (.leaf .fp32 0))) (memory r) off else memory r off
  else memory

/-- Raw independent main-grid recurrence followed by the causal zero fill. -/
def kernelMemory (h : Host.Call) (chosen : Host.Config)
    (memory : RegionName → Nat → Cell Expr) :=
  if h.shape.outputSize=0 then memory else
    clearMemory h (grid h chosen (FPGrid.allPrograms h chosen) memory)

/-- Independent numerical reference for one admitted grid invocation and its
causal cleanup. It contains no source evaluator or final source-state readback. -/
def completeMemory (p : Metadata) (chosen : Host.Config)
    (memory : RegionName → Nat → Cell Expr) :=
  kernelMemory (call p) chosen memory

def resultGraph (p : Metadata) (chosen : Host.Config)
    (initialMemory : RegionName → Nat → Cell Expr) (off : Nat) : Expr :=
  toFloat .fp32 ((completeMemory p chosen initialMemory (call p).Out off).read .fp32)

#specNonCircular kernelMemory avoiding [Kernels.source,FPGrid.schedule,DirectFP.execute,DirectFP.finalState]
#specNonCircular completeMemory avoiding [Kernels.source,FPGrid.schedule,DirectFP.execute,DirectFP.finalState]
#specNonCircular resultGraph avoiding [Kernels.source,FPGrid.schedule,DirectFP.execute,DirectFP.finalState]
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeReference
