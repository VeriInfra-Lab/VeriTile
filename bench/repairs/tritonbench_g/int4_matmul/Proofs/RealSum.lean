import bench.repairs.tritonbench_g.int4_matmul.Spec
namespace VeriTile.Bench.Repairs.Int4Matmul.RealSum
set_option maxHeartbeats 4000000
theorem sum_congr (f g : Nat → ℝ) (n : Nat) (h : ∀ k, k<n → f k=g k) :
    sumReal f n=sumReal g n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [sumReal, ih (fun k hk => h k (by omega)), h n (by omega)]

theorem fin_fold (f : Nat → ℝ) (n : Nat) (z : ℝ) :
    (List.finRange n).foldl (fun t i => t+f i.val) z=z+sumReal f n := by
  induction n with
  | zero => simp [sumReal]
  | succ n ih => simp [List.finRange_succ_last, List.foldl_append, List.foldl_map, ih, sumReal, add_assoc]

theorem sum_add (f : Nat → ℝ) (a b : Nat) :
    sumReal f (a+b)=sumReal f a+sumReal (fun k => f (a+k)) b := by
  induction b with
  | zero => simp [sumReal]
  | succ b ih => simp [sumReal, ih, Nat.add_assoc, add_assoc]

theorem sum_padding (f : Nat → ℝ) (K n : Nat) :
    sumReal (fun k => if k<K then f k else 0) n=sumReal f (min K n) := by
  induction n with
  | zero => simp [sumReal]
  | succ n ih =>
    by_cases hn : n<K
    · have hmin : min K n=n := Nat.min_eq_right (by omega)
      have hnext : min K (n+1)=n+1 := Nat.min_eq_right (by omega)
      simp [sumReal, ih, hn, hmin, hnext]
    · have hmin : min K n=K := Nat.min_eq_left (by omega)
      have hnext : min K (n+1)=K := Nat.min_eq_left (by omega)
      simp [sumReal, ih, hn, hmin, hnext]

def chunks (f : Nat → ℝ) (B : Nat) : Nat → ℝ
  | 0 => 0
  | n+1 => chunks f B n+sumReal (fun k => f (n*B+k)) B

theorem chunks_eq (f : Nat → ℝ) (B n : Nat) : chunks f B n=sumReal f (n*B) := by
  induction n with
  | zero => simp [chunks, sumReal]
  | succ n ih => simp only [chunks, ih, Nat.succ_mul, sum_add]


theorem range_fold (f : Nat → ℝ) (n : Nat) (seed : ℝ) :
    (List.range n).foldl (fun acc i => acc+f i) seed=seed+sumReal f n := by
  induction n with
  | zero => simp [sumReal]
  | succ n ih => simp [List.range_succ,List.foldl_append,ih,sumReal,add_assoc]

theorem sum_distrib (f g : Nat → ℝ) (n : Nat) :
    sumReal (fun k => f k+g k) n=sumReal f n+sumReal g n := by
  induction n with
  | zero => simp [sumReal]
  | succ n ih => simp only [sumReal,ih];ring

theorem sum_swap (f : Nat → Nat → ℝ) (a b : Nat) :
    sumReal (fun i => sumReal (f i) b) a=sumReal (fun j => sumReal (fun i => f i j) a) b := by
  induction a with
  | zero =>
    simp only [sumReal]
    induction b with
    | zero => rfl
    | succ b ih => simp [sumReal,←ih]
  | succ a ih => simp only [sumReal,ih, sum_distrib]

theorem sum_chunks (f : Nat → ℝ) (B n : Nat) :
    sumReal (fun j => sumReal (fun i => f (j*B+i)) B) n=sumReal f (n*B) := by
  induction n with
  | zero => simp [sumReal]
  | succ n ih => simp only [sumReal,ih,Nat.succ_mul,sum_add]

#axiomsClean sum_swap
#axiomsClean sum_chunks
end VeriTile.Bench.Repairs.Int4Matmul.RealSum
