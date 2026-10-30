import bench.repairs.tritonbench_g.bmm_chunk_bwd.Kernels
import bench.tritonbench_g.triton_softmax.Proofs.IndexRewrite

namespace VeriTile.Bench.Repairs.BmmChunkBwd.KernelRewrite
open VeriTile Triton FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

def program (k : ComputeKernel) : Program := ⟨k,[]⟩

/-- A genuine source candidate: exact integer commutation in the
head-index calculation. The complete floating program remains in its original context. -/
theorem equivalent (h : HostCall) (c : Tuning) :
    program (Kernels.fp32Source h c) ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      program (Kernels.indexCommuted h c) := by
  equiv_decompose; fp_prove

theorem execution {A : Type} [Inhabited A] (algebra : FP.Structural.Algebra A)
    (h : HostCall) (c : Tuning) (s : FP.Structural.State A) :
    FP.Structural.exec (engine algebra) (Kernels.fp32Source h c) s=
      FP.Structural.exec (engine algebra) (Kernels.indexCommuted h c) s := by
  change Contextual algebra s
    ((Kernels.fp32Source h c).surfaceBody.take 3)
    ((Kernels.fp32Source h c).surfaceBody.drop 4)
    (.assign .nat [] "pid_h" (.alg (.sub .nat .nil (.ref .nat [] "pid_ch")
      (.mul .nat .nil (.ref .nat [] "pid_c") (.constNat h.groupCount)))))
    (.assign .nat [] "pid_h" (.alg (.sub .nat .nil (.ref .nat [] "pid_ch")
      (.mul .nat .nil (.constNat h.groupCount) (.ref .nat [] "pid_c")))))
  fp_prove

/-- The same candidate edit is exact in the actual signed-INT64 word arithmetic. -/
theorem native_product (x y : BitVec 64) : x*y=y*x := BitVec.mul_comm x y

#axiomsClean native_product
#axiomsClean equivalent
#axiomsClean execution
end VeriTile.Bench.Repairs.BmmChunkBwd.KernelRewrite
