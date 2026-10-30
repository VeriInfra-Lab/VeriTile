import bench.repairs.tritonbench_g.log_softmax.Proofs.Address

namespace VeriTile.Bench.Repairs.LogSoftmax.Coordinates
open Address

theorem dimensions (i : Fin (M*N*K)) : 0<M ∧ 0<N ∧ 0<K := by
  have h := i.isLt
  have : M*N*K≠0 := Nat.ne_of_gt (lt_of_le_of_lt (Nat.zero_le _) h)
  simp only [ne_eq,mul_eq_zero,not_or] at this
  omega

def decode (i : Fin (M*N*K)) : Coord M N K :=
  (⟨i.val/K/N, (Nat.div_lt_iff_lt_mul (dimensions i).2.1).mpr
    ((Nat.div_lt_iff_lt_mul (dimensions i).2.2).mpr i.isLt)⟩,
   ⟨i.val/K%N,Nat.mod_lt _ (dimensions i).2.1⟩,
   ⟨i.val%K,Nat.mod_lt _ (dimensions i).2.2⟩)

theorem decode_linear (i : Fin (M*N*K)) : linear (decode i)=i.val := by
  have row := Nat.div_add_mod (i.val/K) N
  have col := Nat.div_add_mod i.val K
  simp only [linear,offset,decode]
  calc
    i.val/K/N*N*K+(i.val/K%N)*K+i.val%K =
        (N*(i.val/K/N)+(i.val/K%N))*K+i.val%K := by ring
    _ = i.val := by rw [row]; simpa only [Nat.mul_comm K] using col

theorem linear_surjective (i : Nat) (hi : i<M*N*K) :
    ∃ q : Coord M N K, linear q=i := ⟨decode ⟨i,hi⟩,decode_linear _⟩

end VeriTile.Bench.Repairs.LogSoftmax.Coordinates
