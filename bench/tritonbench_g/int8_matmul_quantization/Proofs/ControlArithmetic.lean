import VeriTile.Triton.Float.Integer
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.ControlArithmetic
@[simp] theorem conditional_boolean_result (ι : Type) (p : Prop) [Decidable p] :
    (if p then some (fun _ : ι => Bool.true) else some (fun _ : ι => Bool.false)) =
      some (fun _ => decide p) := by
  by_cases h : p <;> simp [h]
@[simp] theorem tdiv_nat (a b : Nat) : (a : Int).tdiv (b : Int) = ((a/b : Nat) : Int) := by
  rw [Int.tdiv_eq_ediv_of_nonneg (Int.natCast_nonneg _)]; rfl
@[simp] theorem tmod_nat (a b : Nat) : (a : Int).tmod (b : Int) = ((a%b : Nat) : Int) := by
  rw [Int.tmod_eq_emod_of_nonneg (Int.natCast_nonneg _)]; rfl
@[simp] theorem cdiv_int (n d : Nat) :
    ((n : Int) + ((d-1 : Nat) : Int)).tdiv (d : Int) = (((n+d-1)/d : Nat) : Int) := by
  cases d with
  | zero => simp
  | succ d => simp only [Nat.succ_sub_one, Nat.add_succ_sub_one, ← Int.natCast_add, tdiv_nat]
@[simp] theorem cdiv_signed_int (n d : Nat) :
    ((n : Int) + ((d : Int)-1)).tdiv (d : Int) = (((n+d-1)/d : Nat) : Int) := by
  cases d with
  | zero => simp
  | succ d =>
    rw [show (n : Int)+((Nat.succ d : Int)-1)=((n+d : Nat) : Int) by omega, tdiv_nat]
    simp only [Nat.add_succ_sub_one]
@[simp] theorem remainder_int (n d : Nat) :
    (n : Int) - ((n/d : Nat) : Int) * (d : Int) = ((n-n/d*d : Nat) : Int) := by
  rw [← Int.natCast_mul, Int.natCast_sub (Nat.div_mul_le_self n d)]
@[simp] theorem cdiv_one (n d : Nat) (hd : 0<d) (hn : 0<n) (h : n≤d) :
    (n+d-1)/d=1 := by
  apply Nat.div_eq_of_lt_le
  · omega
  · omega
/-- Keep signed subtraction until the existing grouped launch proves it nonnegative. -/
theorem signed_group_size (a b c : Nat) (h : b≤a) :
    (if (a : Int)-(b : Int)<(c : Int) then (a : Int)-(b : Int) else (c : Int)) =
      ((min (a-b) c : Nat) : Int) := by
  rw [← Int.natCast_sub h]
  by_cases small : a-b<c
  · simp [Int.ofNat_lt,small,min_eq_left (Nat.le_of_lt small)]
  · simp [Int.ofNat_lt,small,min_eq_right (Nat.le_of_not_gt small)]

@[simp] theorem signed_remaining_lt (x n k width split : Nat) :
    ((x : Int)<(n : Int)-(k : Int)*(width : Int)*(split : Int)) ↔ x<n-k*(width*split) := by
  rw [← Int.natCast_mul, ← Int.natCast_mul]
  simp only [Nat.mul_assoc]
  omega

@[simp] theorem signed_remaining_single_lt (x n k width : Nat) :
    ((x : Int)<(n : Int)-(k : Int)*(width : Int)) ↔ x<n-k*width := by
  simpa only [Nat.mul_one, Int.natCast_one, Int.mul_one] using signed_remaining_lt x n k width 1

end VeriTile.Bench.Repairs.Int8MatmulQuantization.ControlArithmetic
