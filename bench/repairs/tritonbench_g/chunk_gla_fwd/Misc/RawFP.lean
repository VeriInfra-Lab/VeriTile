import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeRuntime
import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.KernelNative

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPEquiv
open VeriTile Triton
open NativeInstructions
noncomputable section

abbrev producer_returned_fp32 := @NativeRuntime.producer_readback
abbrev consumer_returned_fp32 := @NativeRuntime.consumer_readback
abbrev local_numerical_transport := @NativeRuntime.transport

/-- Complete native source execution, output words and frame over raw allocations. -/
theorem kernel_raw_equiv (c : Host.Dimensions) (mode : Admission.Mode)
    (I : ISA (BitVec 32)) (bits : ∀ b,I.bits b=b) (w : KernelProfile.Profile)
    (scale : BitVec 32) (abi : Physical.ABI c) (fresh : KernelRuntime.Fresh mode abi)
    (before : Nat → BitVec 32) :
    ∃ final,KernelNative.execute I mode w scale abi before=some final ∧
      KernelNative.RawPost I mode w scale abi before final :=
  KernelNative.raw_complete I bits mode w scale abi fresh before

#axiomsClean kernel_raw_equiv
#axiomsClean producer_returned_fp32
#axiomsClean consumer_returned_fp32
#axiomsClean local_numerical_transport
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPEquiv
