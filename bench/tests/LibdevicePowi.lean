import VeriTile.Triton.DSL
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.LibdevicePowi
open VeriTile.Triton
set_option maxRecDepth 16000

/-- Native integer exponent two is retained, including the local fp32 scope. -/
def source32 : ComputeKernel := triton {
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), 2)
}

def source64 : ComputeKernel := triton {
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float64), 2)
}

def sourceNegative : ComputeKernel := triton {
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), -2)
}

example : source32.surfaceBody =
    [.assign .real [] "y" (.compute (.alg .fp32
      (.libdevicePowi .nil (.castFloat .real .real (.const 2.0)) (.constInt 2))))] := rfl

example : source64.surfaceBody =
    [.assign .real [] "y" (.compute (.alg .fp64
      (.libdevicePowi .nil (.castFloat .real .real (.const 2.0)) (.constInt 2))))] := rfl

example : sourceNegative.surfaceBody =
    [.assign .real [] "y" (.compute (.alg .fp32
      (.libdevicePowi .nil (.castFloat .real .real (.const 2.0)) (.constInt (-2)))))] := rfl

/-- error: tl.extra.cuda.libdevice.pow: integer exponent must fit the native int32 overload -/
#guard_msgs in
example : ComputeKernel := triton {
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), 2147483648)
}

/-- error: tl.extra.cuda.libdevice.pow: dynamic integer exponent needs retained int32 width; this source dispatch is not modeled yet -/
#guard_msgs in
example (n : Nat) : ComputeKernel := triton {
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), $(n))
}

/-- error: tl.extra.cuda.libdevice.pow: integer-power overload requires an fp32 or fp64 base; cast the source operand explicitly -/
#guard_msgs in
example : ComputeKernel := triton {
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float16), 2)
}

/-- error: tl.extra.cuda.libdevice.pow: integer-power overload requires an fp32 or fp64 base; cast the source operand explicitly -/
#guard_msgs in
example (n : Nat) : ComputeKernel := triton {
  y = tl.extra.cuda.libdevice.pow($(n), 2)
}

namespace Backend
open FP.Structural

/-- Each native operation returns a deliberately distinct marker. This model
cannot accidentally validate a powi-to-square or powi-to-floating-pow rewrite. -/
private def model : Algebra Int where
  literal _ _ _ := 3
  negInf := 0
  binary _ _ op _ _ := match op with
    | .mul => 11
    | .pow => 17
    | _ => 0
  unary _ _ _ := 0
  libdevicePowi p := some (fun a n => a + n + if p = some .fp32 then 100 else 200)
  cast _ _ _ a := a
  fromNat _ n := n
  fromInt _ n := n
  fp32Bits _ := 0
  fp32Load := id
  reduceMax _ {_} _ _ _ := fun _ => 0
  reduceSum _ {_} _ _ _ := fun _ => 0

private def initial : State Int where
  mem _ _ := .mk .real 0
  regs _ _ _ := none
  pids _ := 0
  numPids _ := 1
  undef d _ _ := defaultValue d

example : evalOp model (some .fp32)
    (.libdevicePowi .nil (.const 2) (.constInt 2)) initial =
      some (fun _ => 105) := rfl

example : evalOp model (some .fp64)
    (.libdevicePowi .nil (.const 2) (.constInt (-2))) initial =
      some (fun _ => 201) := rfl

example : evalOp model (some .fp32)
    (.mul .real .nil (.const 2) (.const 2)) initial = some (fun _ => 11) := rfl

example : evalOp model (some .fp32)
    (.pow .nil (.const 2) (.const 2)) initial = some (fun _ => 17) := rfl

/-- Default precision resolution reaches the integer-power callback. -/
example : evalOp (model.withDefaultPrecision .fp32) none
    (.libdevicePowi .nil (.const 2) (.constInt 2)) initial =
      some (fun _ => 105) := rfl

/-- A local fp64 scope overrides the default profile without changing exponent. -/
example : evalOp (model.withDefaultPrecision .fp32) none
    (.withPrecision .real .fp64 (.libdevicePowi .nil (.const 2) (.constInt 2))) initial =
      some (fun _ => (205 : Int)) := rfl

/-- An unsupported backend fails even though multiplication and floating power
are available in the same algebra. -/
example : evalOp { model with libdevicePowi := fun _ => none } (some .fp32)
    (.libdevicePowi .nil (.const 2) (.constInt 2)) initial = none := rfl

/-- The surface program actually succeeds and writes its powi result register. -/
example : (run (model.withDefaultPrecision .fp32) source32.surfaceBody initial).bind
    (fun s => (s.regs .real [] "y").map (fun v => v PUnit.unit)) = some 105 := by
  simp [source32, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, model, initial, Algebra.withDefaultPrecision, resolvePrecision,
    State.setReg, bop, ofFloat, toFloat]

end Backend

private theorem eval_int (n : Int) (s : BlockState) :
    evalOp (.constInt n) s = some (Tile.scalar n) := by simp only [evalOp]

example (s : BlockState) :
    evalOp (.libdevicePowi .nil (.const (-2)) (.constInt 2)) s =
      some (Tile.scalar (some 4)) := by
  rw [evalOp_libdevicePowi, evalOp_const, eval_int]
  apply congrArg some
  apply Tile.ext
  intro i
  norm_num [Tile.libdevicePowi_data, Tile.scalar_data, WithBot.realPowi, WithBot.map]

example (s : BlockState) :
    evalOp (.libdevicePowi .nil (.const 2) (.constInt (-2))) s =
      some (Tile.scalar (some (1 / 4))) := by
  rw [evalOp_libdevicePowi, evalOp_const, eval_int]
  apply congrArg some
  apply Tile.ext
  intro i
  norm_num [Tile.libdevicePowi_data, Tile.scalar_data, WithBot.realPowi, WithBot.map]

example (x : Op .real a) (n : Op .int b) (bc : Broadcast a b sh) :
    (Op.libdevicePowi bc x n).eraseDType =
      Op.libdevicePowi bc x.eraseDType n.eraseDType := by
  rw [Op.eraseDType.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.LibdevicePowi
