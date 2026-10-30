import bench.repairs.tritonbench_g.triton_argmax.revision004.Proofs.SelectedEmission

/-! Machine lane-election operations. Native lane operands and literals are
i32, bitwise-and and logical shifts execute on BitVec 32, and comparisons
read their signed values. Bounds are local decoded operand/type obligations,
not input shape or launch-family limits. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Revision004.LaneIR
open SelectedEmission

def eval (lane : Nat) : LaneExpr → BitVec 32
  | .lane => BitVec.ofNat 32 lane
  | .literal value => BitVec.ofNat 32 value
  | .bitAnd a b => eval lane a &&& eval lane b
  | .shiftRight a shift => eval lane a >>> shift

def Valid (lane : Nat) : LaneExpr → Prop
  | .lane => lane<2^31
  | .literal value => value<2^31
  | .bitAnd a b => Valid lane a ∧ Valid lane b
  | .shiftRight a shift => Valid lane a ∧ shift<32

theorem eval_exact (expression : LaneExpr) (lane : Nat) (valid : Valid lane expression) :
    (eval lane expression).toNat=expression.eval lane ∧ expression.eval lane<2^31 := by
  induction expression with
  | lane =>
    change lane<2^31 at valid
    exact ⟨Nat.mod_eq_of_lt (by omega : lane<2^32),valid⟩
  | literal value =>
    change value<2^31 at valid
    exact ⟨Nat.mod_eq_of_lt (by omega : value<2^32),valid⟩
  | bitAnd a b ihA ihB =>
    obtain ⟨aValue,aSmall⟩ := ihA valid.1
    obtain ⟨bValue,_⟩ := ihB valid.2
    exact ⟨by simp only [eval,LaneExpr.eval,BitVec.toNat_and,aValue,bValue],Nat.and_le_left.trans_lt aSmall⟩
  | shiftRight a shift ih =>
    obtain ⟨value,small⟩ := ih valid.1
    exact ⟨by simp only [eval,LaneExpr.eval,BitVec.toNat_ushiftRight,value,Nat.shiftRight_eq_div_pow],
      (Nat.div_le_self _ _).trans_lt small⟩

theorem signed_exact (expression : LaneExpr) (lane : Nat) (valid : Valid lane expression) :
    (eval lane expression).toInt=Int.ofNat (expression.eval lane) := by
  have facts := eval_exact expression lane valid
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [facts.1]; have small:=facts.2; omega),facts.1]
  rfl

def guard (lane : Nat) : LaneGuard → Bool
  | .always => Bool.true
  | .equal a b => decide ((eval lane a).toInt=(eval lane b).toInt)
  | .less a b => decide ((eval lane a).toInt<(eval lane b).toInt)
  | .both a b => guard lane a && guard lane b
  | .either a b => guard lane a || guard lane b

def GuardValid (lane : Nat) : LaneGuard → Prop
  | .always => True
  | .equal a b | .less a b => Valid lane a ∧ Valid lane b
  | .both a b | .either a b => GuardValid lane a ∧ GuardValid lane b

theorem guard_exact (condition : LaneGuard) (lane : Nat) (valid : GuardValid lane condition) :
    guard lane condition=condition.eval lane := by
  induction condition with
  | always => rfl
  | equal a b =>
    simp only [guard,LaneGuard.eval,signed_exact a lane valid.1,signed_exact b lane valid.2]
    have same : (Int.ofNat (a.eval lane)=Int.ofNat (b.eval lane)) ↔ a.eval lane=b.eval lane :=
      ⟨Int.ofNat.inj,congrArg Int.ofNat⟩
    simp only [same]
  | less a b =>
    simp only [guard,LaneGuard.eval,signed_exact a lane valid.1,signed_exact b lane valid.2]
    congr 1
    exact propext Int.ofNat_lt
  | both a b ihA ihB => exact congrArg₂ (· && ·) (ihA valid.1) (ihB valid.2)
  | either a b ihA ihB => exact congrArg₂ (· || ·) (ihA valid.1) (ihB valid.2)

#axiomsClean eval_exact
#axiomsClean signed_exact
#axiomsClean guard_exact
end VeriTile.Bench.Repairs.TritonArgmax.Revision004.LaneIR
