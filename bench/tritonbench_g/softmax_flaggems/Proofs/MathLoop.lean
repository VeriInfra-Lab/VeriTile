import bench.tritonbench_g.softmax_flaggems.Proofs.FPLoop
import bench.tritonbench_g.softmax_flaggems.Proofs.ExtendedModel

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.MathLoop
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
open FPLoop
noncomputable section

theorem before (N B t : Nat) (hB : 0<B) : t*B<N ↔ t<RealAddress.rounds N B := by
  by_cases hN : N=0
  · subst N
    have low : B-1<B := by omega
    simp [RealAddress.rounds,Nat.div_eq_of_lt low]
  · exact RealAddress.tile_start_lt_iff (by omega) hB

theorem rounds_mul (c B : Nat) (hB : 0<B) : RealAddress.rounds (c*B) B=c := by
  have le : RealAddress.rounds (c*B) B≤c := by
    have h := before (c*B) B c hB
    omega
  have ge : c≤RealAddress.rounds (c*B) B := by
    by_contra h
    have lt : RealAddress.rounds (c*B) B<c := by omega
    have prod := Nat.mul_lt_mul_of_pos_right lt hB
    have bad := (before (c*B) B (RealAddress.rounds (c*B) B) hB).mp prod
    omega
  omega

theorem walk_exact {β : Type} (f : Nat → β → β) (v : Nat → β) (N B t : Nat)
    (hB : 0<B) (ht : t≤RealAddress.rounds N B)
    (step : ∀ c, c<RealAddress.rounds N B → f (c*B) (v c)=v (c+1)) :
    (walk f N B (t*B) (v t)).2 = v (RealAddress.rounds N B) := by
  by_cases hc : t<RealAddress.rounds N B
  · rw [walk,if_neg (by omega),if_pos ((before N B t hB).mpr hc),step t hc]
    have addr : t*B+B=(t+1)*B := by simp [Nat.add_mul]
    rw [addr]
    exact walk_exact f v N B (t+1) hB (by omega) step
  · have eq : t=RealAddress.rounds N B := by omega
    rw [walk,if_neg (by omega),if_neg (fun h => hc ((before N B t hB).mp h))]
    exact congrArg v eq
termination_by RealAddress.rounds N B-t

#axiomsClean walk_exact
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.MathLoop
