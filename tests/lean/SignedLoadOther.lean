import VeriTile.Triton.DSL
import VeriTile.Triton.Float.SignedCast
import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

/-! Implicit signed load defaults retain the destination-width conversion.
These checks include out-of-range values and both sides of the load mask. -/
namespace VeriTile.Tests.SignedLoadOther
open VeriTile.Triton

private def input : RegionName := "input"

def scalar (value : Int) (enabled : Bool) : ComputeKernel := triton {
  tl.region input = tl.int32
  result = tl.load(input, mask=$((enabled : Bool)), other=$((value : Int)))
}

def read (value : Int) (enabled : Bool) : Op .int [] :=
  .load .int (.region "input" (.constNat 0))
    (.maskOther (.constBool enabled) (.constInt (BitVec.ofInt 32 value).toInt)) .zero

theorem scalar_ast (value : Int) (enabled : Bool) :
    (scalar value enabled).surfaceBody =
      [.assign .int [] "result" (.alg (read value enabled))] := rfl

/-- The frontend's pure-constant fold equals the retained runtime cast. -/
theorem folded_default_eq (value : Int) (s : BlockState) :
    evalOp (Op.intToSigned 32 (.constInt value)) s =
      evalOp (.constInt (BitVec.ofInt 32 value).toInt) s := by
  rw [eval_intToSigned]
  simp [evalOp]

theorem read_value (value : Int) (enabled : Bool) (s : BlockState) :
    evalOp (read value enabled) s = some (Tile.scalar
      (if enabled then s.readMemValue .int "input" 0
       else (BitVec.ofInt 32 value).toInt)) := by
  cases enabled <;> simp [read, evalOp]

theorem masked_sign_boundary (s : BlockState) :
    evalOp (read (2^31) Bool.false) s = some (Tile.scalar (-2147483648)) :=
  read_value (2^31) Bool.false s

theorem masked_wrap (s : BlockState) :
    evalOp (read (2^32) Bool.false) s = some (Tile.scalar 0) :=
  read_value (2^32) Bool.false s

theorem active_load (value : Int) (s : BlockState) :
    evalOp (read value Bool.true) s = some (Tile.scalar (s.readMemValue .int "input" 0)) :=
  read_value value Bool.true s

def named (value : Int) : ComputeKernel := triton {
  tl.region input = tl.int32
  wide = $((value : Int))
  result = tl.load(input, mask=$((Bool.false : Bool)), other=wide)
}

theorem named_default_is_narrowed (value : Int) :
    (named value).surfaceBody.getLast? = some
      (.assign .int [] "result" (.alg (.load .int (.region "input" (.constNat 0))
        (.maskOther (.constBool Bool.false) (.intToSigned 32 (.ref .int [] "wide"))) .zero))) := rfl

def byte (value : Int) : ComputeKernel := triton {
  tl.region input = tl.int8
  result = tl.load(input, mask=$((Bool.false : Bool)), other=$((value : Int)))
}

theorem byte_default_is_narrowed (value : Int) :
    (byte value).surfaceBody =
      [.assign .int [] "result" (.alg (.load .int (.region "input" (.constNat 0))
        (.maskOther (.constBool Bool.false) (.constInt (BitVec.ofInt 8 value).toInt)) .zero))] := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.SignedLoadOther
