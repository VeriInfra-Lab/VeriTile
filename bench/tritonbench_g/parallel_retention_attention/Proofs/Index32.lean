import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Core.SignedCast
import VeriTile.Triton.Float.FixedInteger
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.Index32
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000

@[irreducible] def signed (value : Int) : Int :=
  let bits:=Int.toNat (value % 4294967296) % 4294967296
  if 2*bits<4294967296 then (bits : Int) else (bits : Int)-4294967296

theorem expansion {α : Type} (value : Int) :
    (if 2*integralMod (α := α) .nat (Int.toNat (integralMod (α := α) .int value 4294967296)) 4294967296<4294967296
     then (integralMod (α := α) .nat (Int.toNat (integralMod (α := α) .int value 4294967296)) 4294967296 : Int)
     else (integralMod (α := α) .nat (Int.toNat (integralMod (α := α) .int value 4294967296)) 4294967296 : Int)-4294967296)=signed value:=by unfold signed; rfl

/-- Preserve the signed narrowing node while exposing its exact word value.
The lemma concerns integer evaluation only, for every floating backend. -/
@[simp 2000] theorem eval_signed {α : Type} [Inhabited α] (M : Algebra α)
    (precision : Option ComputeDType) (value : Op .int shape) (s : State α) :
    evalOp M precision (Op.intToSigned 32 value) s=(do
      let values ← evalOp M precision value s
      pure (fun i=>signed (values i))) := by
  change evalOp M precision (Op.unsignedToInt 32 (Op.intToUnsigned 32 value)) s = _
  rw [FP.Structural.eval_unsignedToInt,FP.Structural.eval_intToUnsigned]
  cases h : evalOp M precision value s <;>
    simp only [h,bind,pure,Option.bind_none,Option.bind_some]
  apply congrArg some
  funext i
  simp only [IntegerExpr.signedBits,BitVec.toNat_ofInt]
  unfold signed
  rfl

@[simp] theorem nat_negative (n : Nat) : ¬(n : Int)<0 := by omega

theorem nat_remainder (n d : Nat) :
    (n : Int)-(d : Int)*((n : Int)/(d : Int))=((n%d : Nat) : Int) := by
  have equation:=congrArg (fun x : Nat=>(x : Int)) (Nat.mod_add_div n d)
  simp only [Int.natCast_add,Int.natCast_mul,Int.natCast_ediv] at equation
  omega

@[simp] theorem natural_product (x y : Nat) : Int.toNat ((x : Int)*y)=x*y := by
  rw [←Int.natCast_mul]
  rfl

theorem nat_value (n : Nat) (bound : n<2147483648) : signed (n : Int)=(n : Int) := by
  have hn : n<4294967296:=by omega
  have hi : (n : Int)<4294967296:=by omega
  have twice : 2*n<4294967296:=by omega
  simp [signed,Int.emod_eq_of_lt (Int.natCast_nonneg n) hi,Nat.mod_eq_of_lt hn,twice]

theorem masked {α : Type} (row column : Nat) (bound : row<2147483648) (f : Int → α) (zero : α) :
    (if column≤row then f (signed row-signed column) else zero)=
      if column≤row then f ((row : Int)-column) else zero := by
  by_cases active : column≤row
  · rw [if_pos active,if_pos active,nat_value row bound,nat_value column (by omega)]
  · simp only [active,if_false]

#axiomsClean eval_signed
#axiomsClean nat_value
#axiomsClean masked
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.Index32
