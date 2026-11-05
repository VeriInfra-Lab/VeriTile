import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeSourceFP

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeFP
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
section Cumulative
variable (dtype : FloatDType) (input o : RegionName)
    (head time feature T S BT BS BH base : Nat)
def cumulative : Program where
  kernel := NativeSource.cumOriginal dtype .fp32 input o head time feature T S BT BS BH base
  domain := []
def cumulativeIndexFirst : Program :=
  { cumulative dtype input o head time feature T S BT BS BH base with
    kernel := NativeSource.cumIndexFirst dtype .fp32 input o head time feature T S BT BS BH base }

theorem cumulative_equiv :
    cumulative dtype input o head time feature T S BT BS BH base
      ≡[([] : Spec.Assumptions FP.GuardedFragment)]
    cumulativeIndexFirst dtype input o head time feature T S BT BS BH base := by
  exact CumulativeSourceFP.equivalent dtype .fp32 input o head time feature T S BT BS BH base
end Cumulative

#axiomsClean cumulative_equiv
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeFP
