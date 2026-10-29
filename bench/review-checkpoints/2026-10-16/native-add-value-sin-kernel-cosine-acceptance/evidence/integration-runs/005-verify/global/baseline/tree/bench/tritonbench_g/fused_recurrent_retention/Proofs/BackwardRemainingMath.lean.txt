import bench.tritonbench_g.fused_recurrent_retention.Proofs.MaskedMath
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Addressing

/-! Independent reverse-loop recurrence. At entry with r rows remaining, the
source carry equals b times the suffix at r. At row r-1, += forms that row's
suffix before gradients are reduced; *=b prepares the next carry. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardRemainingMath
open VeriTile.Triton MaskedMath
noncomputable section

def carry (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (remaining : Nat) (k : Fin p.BK) (j : Fin p.BV) : ℝ :=
  decay p l * dStateValue p l initial r.q r.do_ remaining k j

theorem at_end (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (k : Fin p.BK) (j : Fin p.BV) : carry p l initial r p.T k j = 0 := by
  simp [carry,dstate_end]

theorem current (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (ht : t < p.T) (k : Fin p.BK) (j : Fin p.BV) :
    carry p l initial r (t+1) k j +
      (keyValue p l initial r.q t k*p.scale)*valueValue p l initial r.do_ t j =
      dStateValue p l initial r.q r.do_ t k j := by
  rw [dstate_step p l initial r.q r.do_ t ht]
  simp only [carry]
  ring

theorem next (p : Layout) (l : Launch) (initial : BlockState) (r : BackwardRegions)
    (t : Nat) (k : Fin p.BK) (j : Fin p.BV) :
    dStateValue p l initial r.q r.do_ t k j * decay p l = carry p l initial r t k j := by
  unfold carry
  ring

/-- Only a positive remaining row implies the decremented pointer will be
read again. No equation is imposed on the terminal pointer value. -/
theorem pointer_live (base stride t : Nat) (ht : 0 < t) :
    (base+t*stride)-stride = base+(t-1)*stride := by
  have he : t = (t-1)+1 := by omega
  conv_lhs => rw [he]
  exact Addressing.subtract_stride_live base stride (t-1)

#axiomsClean at_end
#axiomsClean current
#axiomsClean next
#axiomsClean pointer_live
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardRemainingMath
