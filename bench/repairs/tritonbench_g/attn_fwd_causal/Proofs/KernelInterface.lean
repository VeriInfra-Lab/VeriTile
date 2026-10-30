import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Kernel

namespace VeriTile.Bench.Repairs.AttnFwdCausal.RawKernel
open VeriTile Triton FP.Structural Source Whole

/-- Finite regional bounds follow from kernel geometry. This witness contains
only capacities, active-address bounds and region separation, with no Tensor API. -/
def storage (p : Parameters)
    (apart : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS) :
    TensorStorage p where
  sizeQ := strideEnvelope p p.q + 1
  sizeK := strideEnvelope p p.k + 1
  sizeV := strideEnvelope p p.v + 1
  sizeQS := p.batches*p.heads*((p.length+127)/128)
  sizeKS := p.batches*p.heads*((p.length+63)/64)
  sizeOut := strideEnvelope p p.o + 1
  query := by
    intro b hb h hh n hn d hd
    exact Nat.lt_succ_of_le (formed_offset_le p p.q b h n d (by omega) (by omega) (by omega) (by omega))
  key := by
    intro b hb h hh n hn d hd
    exact Nat.lt_succ_of_le (formed_offset_le p p.k b h n d (by omega) (by omega) (by omega) (by omega))
  value := by
    intro b hb h hh n hn d hd
    exact Nat.lt_succ_of_le (formed_offset_le p p.v b h n d (by omega) (by omega) (by omega) (by omega))
  scales := ⟨rfl,rfl⟩
  output := by
    intro b hb h hh n hn d hd
    exact Nat.lt_succ_of_le (formed_offset_le p p.o b h n d (by omega) (by omega) (by omega) (by omega))
  separated := apart

end VeriTile.Bench.Repairs.AttnFwdCausal.RawKernel
