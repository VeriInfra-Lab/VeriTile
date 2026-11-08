import VeriTile.Triton.Float.Integer
namespace VeriTile.Bench.Repairs.BmmChunkFwd.ControlArithmetic
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
end VeriTile.Bench.Repairs.BmmChunkFwd.ControlArithmetic

namespace VeriTile.Triton.FP.Structural
/-- The signed source loop with natural mathematical bounds. Its register
remains signed; only the proof's iteration measure uses natural numbers. -/
theorem rangeInt_nat_step {α : Type} [Inhabited α] (M : Algebra α)
    (name : RegName) (cur stop stride : Nat) (body : List ComputeStmt) (s : State α) :
    rangeInt M name (cur : Int) (stop : Int) stride body s =
      if stride=0 then some s else if cur<stop then do
        let t ← run M body (s.setReg name .int [] (fun _ => (cur : Int)))
        rangeInt M name ((cur+stride : Nat) : Int) (stop : Int) stride body t
      else some s := by
  rw [rangeInt]
  simp only [Int.ofNat_lt, Int.natCast_add]
end VeriTile.Triton.FP.Structural
