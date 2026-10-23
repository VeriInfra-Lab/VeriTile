import bench.repairs.tritonbench_g.bmm_chunk_fwd.Kernels
import bench.tritonbench_g.triton_softmax.Proofs.IndexRewrite

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
      program (Kernels.indexCommuted h c) := by
  equiv_decompose; fp_prove

theorem execution {A : Type} [Inhabited A] (algebra : FP.Structural.Algebra A)
    (h : Host.Call) (c : Host.Config) (s : FP.Structural.State A) :
    FP.Structural.exec (engine algebra) (Kernels.fp32Source h c) s=
      FP.Structural.exec (engine algebra) (Kernels.indexCommuted h c) s := by
  change Contextual algebra s
    ((Kernels.fp32Source h c).surfaceBody.take 3)
    ((Kernels.fp32Source h c).surfaceBody.drop 4)
    (.assign .nat [] "pid_h" (.alg (.sub .nat .nil (.ref .nat [] "pid_ch")
      (.mul .nat .nil (.ref .nat [] "pid_c") (.constNat h.shape.groups)))))
    (.assign .nat [] "pid_h" (.alg (.sub .nat .nil (.ref .nat [] "pid_ch")
      (.mul .nat .nil (.constNat h.shape.groups) (.ref .nat [] "pid_c")))))
  fp_prove

#axiomsClean equivalent
#axiomsClean execution
end VeriTile.Bench.Repairs.BmmChunkFwd.KernelRewrite
