import VeriTile.Triton.DSL
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.EvalOpR
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.LibdevicePowiDynamic
open VeriTile.Triton
set_option maxRecDepth 16000

/-- A declared int32 load retains native overload provenance through a copy
and a same-width cast. The integer exponent is never converted to floating. -/
def source (X : RegionName) : ComputeKernel := triton {
  tl.region X = tl.int32
  n = tl.load(X + $(0))
  copy = n.to(tl.int32)
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), copy)
}

example (X : RegionName) : (source X).surfaceBody =
    [.assign .int [] "n" (.alg (.load .int (.region X (.constNat 0)) .none .zero)),
     .assign .int [] "copy" (.alg (.ref .int [] "n")),
     .assign .real [] "y" (.compute (.alg .fp32
       (.libdevicePowi .nil (.castFloat .real .real (.const 2.0))
         (.ref .int [] "copy"))))] := rfl

/-- Nested loads and block-pointer loads retain the same declared width. -/
def nested (X : RegionName) : ComputeKernel := triton {
  tl.region X = tl.int32
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), tl.load(X + $(0)))
}

def block (X : RegionName) : ComputeKernel := triton {
  tl.region X = tl.int32
  p = tl.make_block_ptr(base=X, shape=($(3)), strides=($(1)),
    offsets=($(0)), block_shape=($(4)), order=(0))
  n = tl.load(p, boundary_check=([0] : List Nat))
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), n)
}

/-- error: tl.extra.cuda.libdevice.pow: dynamic integer exponent needs retained int32 width; this source dispatch is not modeled yet -/
#guard_msgs in
example (X : RegionName) : ComputeKernel := triton {
  tl.region X = tl.int64
  n = tl.load(X + $(0))
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), n)
}

/- An int32 source cast to int64 cannot keep its original native dispatch. -/
/-- error: tl.extra.cuda.libdevice.pow: dynamic integer exponent needs retained int32 width; this source dispatch is not modeled yet -/
#guard_msgs in
example (X : RegionName) : ComputeKernel := triton {
  tl.region X = tl.int32
  n = tl.load(X + $(0))
  wide = n.to(tl.int64)
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), wide)
}

/-- error: tl.extra.cuda.libdevice.pow: dynamic integer exponent needs retained int32 width; this source dispatch is not modeled yet -/
#guard_msgs in
example (X : RegionName) : ComputeKernel := triton {
  tl.region X = tl.uint32
  n = tl.load(X + $(0))
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), n)
}

/- Native extern lookup occurs before generic floating promotion. -/
/-- error: tl.extra.cuda.libdevice.pow: native floating-power overload requires matching fp32 or fp64 operands; cast the source operands explicitly -/
#guard_msgs in
example : ComputeKernel := triton {
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), (3.0).to(tl.float64))
}

/-- error: tl.extra.cuda.libdevice.pow: native floating-power overload requires matching fp32 or fp64 operands; cast the source operands explicitly -/
#guard_msgs in
example : ComputeKernel := triton {
  y = tl.extra.cuda.libdevice.pow((2.0).to(tl.float32), (3.0).to(tl.float16))
}

open FP.Structural in
private def model : Algebra Int where
  literal _ _ _ := 2
  negInf := 0
  binary _ _ _ _ _ := 17
  unary _ _ _ := 0
  libdevicePowi p := some (fun a n => a + n + if p = some .fp32 then 100 else 200)
  cast _ _ _ a := a
  fromNat _ n := n
  fromInt _ n := n
  fp32Bits _ := 0
  fp32Load := id
  reduceMax _ {_} _ _ _ := fun _ => 0
  reduceSum _ {_} _ _ _ := fun _ => 0

open FP.Structural in
private def initial (n : Int) : State Int where
  mem _ _ := .mk .int n
  regs _ _ _ := none
  pids _ := 0
  numPids _ := 1
  undef d _ _ := defaultValue d

open FP.Structural in
example : (run model (source "X").surfaceBody (initial (-3))).bind
    (fun s => (s.regs .real [] "y").map (fun v => v PUnit.unit)) = some 99 := by
  simp [source, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, model, initial, State.setReg, bop, ofFloat, toFloat,
    Int32Bits.valuesValid, Int32Bits.inRange, TileShape.allIndices]

open FP.Structural in
example : evalOp model (some .fp32)
    (.libdevicePowi .nil (.const 2) (.constInt (-2147483648))) (initial 0) =
      some (fun _ => (-2147483546 : Int)) := rfl

open FP.Structural in
example : evalOp model (some .fp32)
    (.libdevicePowi .nil (.const 2) (.constInt 2147483647)) (initial 0) =
      some (fun _ => (2147483749 : Int)) := rfl

open FP.Structural in
example : evalOp model (some .fp32)
    (.libdevicePowi .nil (.const 2) (.constInt 2147483648)) (initial 0) = none := rfl

open FP.Structural in
example : evalOp model (some .fp32)
    (.libdevicePowi .nil (.const 2) (.constInt (-2147483649))) (initial 0) = none := rfl

open FP.Structural in
example : run model (source "X").surfaceBody (initial 2147483648) = none := by
  simp [source, ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, model, initial, State.setReg, bop, ofFloat, toFloat,
    Int32Bits.valuesValid, Int32Bits.inRange, TileShape.allIndices]

open FP.Structural in
example : evalOp { model with libdevicePowi := fun _ => none } (some .fp32)
    (.libdevicePowi .nil (.const 2) (.constInt 2)) (initial 0) = none := rfl

open FP.Structural in
example : evalOp model (some .fp32)
    (.libdevicePowi .nil (.const 2) (.ref .int [] "missing")) (initial 0) = none := rfl

/-- Exact and rounded interpreters reject the same invalid typed exponents. -/
example (s : BlockState) :
    evalOp (.libdevicePowi .nil (.const 2) (.constInt 2147483648)) s = none := by
  simp [evalOp, Int32Bits.valuesValid, Int32Bits.inRange, Tile.scalar,
    -Tile.scalar_eta, TileShape.allIndices]

example (R : RoundingModel) (s : BlockState) :
    evalOpR R (.libdevicePowi .nil (.const 2) (.constInt (-2147483649))) s = none := by
  simp [evalOpR, Int32Bits.valuesValid, Int32Bits.inRange, Tile.scalar,
    -Tile.scalar_eta, TileShape.allIndices]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.LibdevicePowiDynamic
