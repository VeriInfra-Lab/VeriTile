import bench.tritonbench_g.chunk_gated_attention.SourceDefinitions
import bench.tritonbench_g.chunk_gated_attention.Sources.RealSource

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
open VeriTile Triton

theorem inner_real_source (r : Admission.Inner) (base : Nat) :
    RebasedProgram.source (RealSource.layout r) RealSource.regions .fp32 .fp32 .fp32 .fp32 .fp32
      base r.gateK r.seed.isSome r.final.isSome Bool.false = innerKernel r base := by rfl

theorem prefix_real_source (r : Admission.Prefix) (BS base : Nat) :
    NativeSource.cumOriginal .fp32 .fp32 "s" "o"
      ((HostFP.padded r.T r.BT)*(HostFP.padded r.S 64)) (HostFP.padded r.S 64) 1
      (HostFP.padded r.T r.BT) (HostFP.padded r.S 64) r.BT BS (r.B*r.H) base =
      prefixKernel r BS base := by rfl

#axiomsClean inner_real_source
#axiomsClean prefix_real_source
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly
