import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Support.KernelEntryAdapters
import bench.tritonbench_g.triton_softmax.Proofs.Mathematical_IndexRewrite

namespace VeriTile.Bench.Repairs.BmmChunkFwd.KernelRewrite
open VeriTile Triton FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

def program (k : ComputeKernel) : Program := ⟨k,[]⟩

/-- A genuine source candidate: exact integer commutation in the head index.
The complete floating program remains in its original context. -/
theorem equivalent (h : Host.Call) (c : Host.Config) :
    program (Kernels.fp32Source h c) ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      program (Kernels.indexCommutedKernel h c) := by
  split_equiv; solve_fp

theorem execution {A : Type} [Inhabited A] (algebra : FP.Structural.Algebra A)
    (h : Host.Call) (c : Host.Config) (s : FP.Structural.State A) :
    FP.Structural.exec (engine algebra) (Kernels.fp32Source h c) s=
      FP.Structural.exec (engine algebra) (Kernels.indexCommutedKernel h c) s := by
  change Contextual algebra s
    ((Kernels.fp32Source h c).surfaceBody.take 22)
    ((Kernels.fp32Source h c).surfaceBody.drop 23)
    (.assign .int [] "pid_h" (.alg (.sub .int .nil (.ref .int [] "pid_ch")
      (.mul .int .nil (.ref .int [] "pid_c") (.castNatToInt (.constNat h.shape.groups))))))
    (.assign .int [] "pid_h" (.alg (.sub .int .nil (.ref .int [] "pid_ch")
      (.mul .int .nil (.castNatToInt (.constNat h.shape.groups)) (.ref .int [] "pid_c")))))
  solve_fp

#axiomsClean equivalent
#axiomsClean execution
end VeriTile.Bench.Repairs.BmmChunkFwd.KernelRewrite
