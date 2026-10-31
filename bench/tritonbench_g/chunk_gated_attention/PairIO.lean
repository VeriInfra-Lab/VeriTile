import bench.tritonbench_g.chunk_gated_attention.Kernels
import bench.tritonbench_g.chunk_gated_attention.Sources.NativeFP

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.SourcePair
open VeriTile Triton FP.ProfiledRewrite
noncomputable section

def innerOriginal (c : HostFP.Inner) (base : Nat) : Program :=
  NativeFP.recurrent .fp32 .fp32 .fp32 .fp32 .fp32 "k" "v" "g" "h" "h0" "ht"
    (c.PT*c.PK) c.PK 1 (c.PT*c.PV) c.PV 1 (c.NT*c.PK*c.PV) c.PV 1
    c.PT c.PK c.PV c.BT c.BK c.BV c.NT base c.gateK c.hasSeed c.storeFinal

def innerSumLeft (c : HostFP.Inner) (base : Nat) : Program :=
  { innerOriginal c base with kernel := innerCandidate c base }

def prefixOriginal (c : HostFP.Cumulative) (base : Nat) : Program :=
  ⟨NativeGrid.prefixSource c base, []⟩

def prefixIndexFirst (c : HostFP.Cumulative) (base : Nat) : Program :=
  ⟨prefixCandidate c base, []⟩

def indexAssumptions : Spec.Assumptions FP.GuardedFragment := []

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.SourcePair
