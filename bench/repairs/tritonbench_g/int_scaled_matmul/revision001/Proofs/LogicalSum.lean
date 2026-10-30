import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.BlockWordReference
namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.LogicalSum
open VeriTile.Triton

def sum (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n+1 => sum f n + f n

theorem sum_congr (f g : Nat → Int) (n : Nat) (h : ∀ k, k<n → f k=g k) :
    sum f n=sum g n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [sum, ih (fun k hk => h k (by omega)), h n (by omega)]

theorem fin_fold (f : Nat → Int) (n : Nat) (z : Int) :
    (List.finRange n).foldl (fun t i => t+f i.val) z=z+sum f n := by
  induction n with
  | zero => simp [sum]
  | succ n ih => simp [List.finRange_succ_last, List.foldl_append, List.foldl_map, ih, sum, Int.add_assoc]

theorem sum_add (f : Nat → Int) (a b : Nat) :
    sum f (a+b)=sum f a+sum (fun k => f (a+k)) b := by
  induction b with
  | zero => simp [sum]
  | succ b ih => simp [sum, ih, Nat.add_assoc, Int.add_assoc]

theorem sum_padding (f : Nat → Int) (K n : Nat) :
    sum (fun k => if k<K then f k else 0) n=sum f (min K n) := by
  induction n with
  | zero => simp [sum]
  | succ n ih =>
    by_cases hn : n<K
    · have hmin : min K n=n := Nat.min_eq_right (by omega)
      have hnext : min K (n+1)=n+1 := Nat.min_eq_right (by omega)
      simp [sum, ih, hn, hmin, hnext]
    · have hmin : min K n=K := Nat.min_eq_left (by omega)
      have hnext : min K (n+1)=K := Nat.min_eq_left (by omega)
      simp [sum, ih, hn, hmin, hnext]

def chunks (f : Nat → Int) (B : Nat) : Nat → Int
  | 0 => 0
  | n+1 => chunks f B n+sum (fun k => f (n*B+k)) B

theorem chunks_eq (f : Nat → Int) (B n : Nat) : chunks f B n=sum f (n*B) := by
  induction n with
  | zero => simp [chunks, sum]
  | succ n ih => simp only [chunks, ih, Nat.succ_mul, sum_add]

theorem ceil_cover (K B : Nat) (hB : 0<B) : K≤((K+B-1)/B)*B := by
  have hq := Nat.div_add_mod (K+B-1) B
  have hr := Nat.mod_lt (K+B-1) hB
  rw [Nat.mul_comm B] at hq
  omega

theorem ceil_exact (K B : Nat) (hB : 0<B) (he : K%B=0) :
    K=((K+B-1)/B)*B := by
  have hq := Nat.div_add_mod K B
  have hc := Nat.div_add_mod (K+B-1) B
  have hr := Nat.mod_lt (K+B-1) hB
  rw [Nat.mul_comm B, he] at hq
  have hk : K=K/B*B := by omega
  have hd : (K+B-1)/B=K/B := by
    apply Nat.div_eq_of_lt_le
    · omega
    · rw [Nat.add_mul, Nat.one_mul]
      omega
  rw [hd]
  exact hk

#axiomsClean sum_congr
#axiomsClean fin_fold
#axiomsClean sum_add
#axiomsClean sum_padding
#axiomsClean chunks_eq
#axiomsClean ceil_cover
#axiomsClean ceil_exact
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.LogicalSum
