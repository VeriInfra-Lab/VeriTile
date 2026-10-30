import bench.repairs.tritonbench_g.chunk_gated_attention.KernelOnly.PairIO

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.SourcePair
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

theorem inner_equivalent (c : HostFP.Inner) (base : Nat) (R : FP.ScalarArithmetic.Rules) :
    innerOriginal c base ≡[R] innerSumLeft c base :=
  NativeFP.recurrent_equiv .fp32 .fp32 .fp32 .fp32 .fp32 "k" "v" "g" "h" "h0" "ht"
    (c.PT*c.PK) c.PK 1 (c.PT*c.PV) c.PV 1 (c.NT*c.PK*c.PV) c.PV 1
    c.PT c.PK c.PV c.BT c.BK c.BV c.NT base c.gateK c.hasSeed c.storeFinal R

theorem prefix_equivalent (c : HostFP.Cumulative) (base : Nat) :
    prefixOriginal c base ≡[indexAssumptions] prefixIndexFirst c base :=
  NativeFP.cumulative_equiv .fp32 "s" "o"
    (c.PT*c.PS) c.PS 1 c.PT c.PS c.BT c.BS c.heads base

#axiomsClean inner_equivalent
#axiomsClean prefix_equivalent
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.SourcePair
