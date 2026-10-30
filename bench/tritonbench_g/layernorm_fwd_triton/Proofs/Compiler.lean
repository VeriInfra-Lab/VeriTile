import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Grid
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.NativeAddress

/-! Local syntax contracts for the actual selected compilation. The object
bytes, ABI, options and graph are one object. A compiler adapter must decode
that object's bytes and establish these local node/operand/control recipes.
There is no result, reduction-value or kernel-execution equation in the
adapter. This is an ordinary compiler boundary, not numerical conformance.
The seven offline profiles are evidence for this boundary, not its full domain. -/
namespace VeriTile.Bench.Repairs.LayernormFwdTriton
namespace Compiler
open Address Memory Execution Native
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
open scoped Classical

inductive Reg where
  | acc | value | mu | inv | weight | eps | total | count | z
  deriving DecidableEq

inductive Expr where
  | reg (r : Reg)
  | zero | one
  | add (a b : Expr) | sub (a b : Expr) | mul (a b : Expr) | div (a b : Expr)
  | fma (a b c : Expr) | sqrt (a : Expr)
  deriving DecidableEq

def eval (ops : Ops α) (env : Reg → α) : Expr → α
  | .reg r => env r
  | .zero => ops.zero
  | .one => ops.one
  | .add a b => ops.add (eval ops env a) (eval ops env b)
  | .sub a b => ops.sub (eval ops env a) (eval ops env b)
  | .mul a b => ops.mul (eval ops env a) (eval ops env b)
  | .div a b => ops.div (eval ops env a) (eval ops env b)
  | .fma a b c => ops.fma (eval ops env a) (eval ops env b) (eval ops env c)
  | .sqrt a => ops.sqrt (eval ops env a)

def orderedExpr (swap : Bool) (f : Expr → Expr → Expr) (a b : Expr) : Expr :=
  if swap then f b a else f a b

def meanExpr (swap : Bool) : Expr := orderedExpr swap .add (.reg .acc) (.reg .value)
def centerExpr : Expr := .sub (.reg .value) (.reg .mu)
def carryExpr : Carry → Expr
  | .fused => .fma (.reg .z) (.reg .z) (.reg .acc)
  | .separate swap => orderedExpr swap .add (.reg .acc) (.mul (.reg .z) (.reg .z))
def divideExpr (erase : Bool) : Expr :=
  if erase then .reg .total else .div (.reg .total) (.reg .count)
def inverseExpr (swap : Bool) : Expr :=
  .div .one (.sqrt (orderedExpr swap .add (.reg .total) (.reg .eps)))
def outputExpr (swap : Bool) : Expr :=
  .mul (orderedExpr swap .mul (.sub (.reg .value) (.reg .mu)) (.reg .inv)) (.reg .weight)

inductive IntReg where
  | seq | head | off | lane | n | xSeq | xHead | xDim | ySeq | yHead | yDim | wHead | wDim
  deriving DecidableEq
inductive IntExpr where
  | widened (r : IntReg)
  | add (a b : IntExpr)
  | mul (a b : IntExpr)
  deriving DecidableEq

def columnExpr : IntExpr := .add (.widened .off) (.widened .lane)
def addressExpr (s h d : IntReg) : IntExpr :=
  .add (.add (.mul (.widened .seq) (.widened s))
    (.mul (.widened .head) (.widened h))) (.mul columnExpr (.widened d))
def weightAddressExpr : IntExpr :=
  .add (.mul (.widened .head) (.widened .wHead)) (.mul columnExpr (.widened .wDim))

/-- The decoder retains operand ordering, integer widening before products,
all three block128 loops, and all live load/store predicates. Loop unrolling and
pointer strength reduction may be decoded to these local recipes using the
separate signed64/masked-pointer lemmas, without a whole-run claim. -/
structure Graph where
  (meanUpdate center varianceUpdate meanDivide varianceDivide inverse output : Expr)
  (meanTree varianceTree : SumTree)
  (xAddress yAddress wAddress : IntExpr)
  (loopStart loopStep laneCount passCount : Nat)
  (meanMasked varianceMasked outputMasked centerMasked : Bool)
  (fp32Loads fp32Accumulator fp32OutputCast : Bool)

/-- Only instruction identities/configuration, not their numerical laws. -/
structure PrimitiveNames where
  (add sub mul fma divide sqrt countConvert epsABI : String)

inductive ScalarABI where
  | i32 | i64 | u1 | u32 | u64 | fp32 | fp64
  deriving DecidableEq
inductive ScalarBinding where
  | runtime (abi : ScalarABI)
  | specialized (value : Int)

structure Signature where
  integerArgs : List ScalarBinding
  integerCount : integerArgs.length=9
  epsilon : ScalarBinding
  pointerWidths : List Nat
  pointers_fp32 : pointerWidths=[32,32,32]
structure Options where
  target : String
  numWarps : Nat
  numStages : Nat
  enableFusion : Bool
  defaults : String
structure Object where
  identity : Nat
  bytes : ByteArray
  sourceHash : String
  signature : Signature
  options : Options
  primitives : PrimitiveNames

/-- These are scalar syntax equalities and tree leaf coverage, never an
assumption that a reduction already has the desired value. `Graph` must come
from decoding the very bytes subsequently admitted and launched. -/
structure LocalRecipes (N : Nat) (g : Graph) where
  (meanSwap epsSwap normalSwap : Bool)
  varianceCarry : Carry
  eraseUnitDivision : Bool
  eraseUnitValid : eraseUnitDivision=Bool.true → N=1
  meanNode : g.meanUpdate=meanExpr meanSwap
  centerNode : g.center=centerExpr
  carryNode : g.varianceUpdate=carryExpr varianceCarry
  meanDivision : g.meanDivide=divideExpr eraseUnitDivision
  varianceDivision : g.varianceDivide=divideExpr eraseUnitDivision
  inverseNode : g.inverse=inverseExpr epsSwap
  outputNode : g.output=outputExpr normalSwap
  meanLeaves : g.meanTree.leaves.Perm (List.finRange 128)
  varianceLeaves : g.varianceTree.leaves.Perm (List.finRange 128)
  xRecipe : g.xAddress=addressExpr .xSeq .xHead .xDim
  yRecipe : g.yAddress=addressExpr .ySeq .yHead .yDim
  wRecipe : g.wAddress=weightAddressExpr
  start : g.loopStart=0
  step : g.loopStep=128
  lanes : g.laneCount=128
  passes : g.passCount=3
  masks : g.meanMasked=Bool.true ∧ g.varianceMasked=Bool.true ∧
    g.outputMasked=Bool.true ∧ g.centerMasked=Bool.true
  casts : g.fp32Loads=Bool.true ∧ g.fp32Accumulator=Bool.true ∧ g.fp32OutputCast=Bool.true

structure Decoding where
  decode : ByteArray → Signature → Options → PrimitiveNames → Graph

def Decoding.inspect (decoder : Decoding) (object : Object) : Graph :=
  decoder.decode object.bytes object.signature object.options object.primitives

structure Selected (decoder : Decoding) (N : Nat) where
  object : Object
  sourceBound : object.sourceHash=candidateSHA256
  recipes : LocalRecipes N (decoder.inspect object)

def Selected.plan (code : Selected decoder N) : Plan N where
  meanTree := (decoder.inspect code.object).meanTree
  varianceTree := (decoder.inspect code.object).varianceTree
  meanComplete := code.recipes.meanLeaves
  varianceComplete := code.recipes.varianceLeaves
  meanSwap := code.recipes.meanSwap
  epsSwap := code.recipes.epsSwap
  normalSwap := code.recipes.normalSwap
  varianceCarry := code.recipes.varianceCarry
  eraseUnitDivision := code.recipes.eraseUnitDivision
  eraseUnitValid := code.recipes.eraseUnitValid

def environment (ops : Ops α) (acc value mu inv weight eps total count z : α) : Reg → α
  | .acc => acc | .value => value | .mu => mu | .inv => inv | .weight => weight
  | .eps => eps | .total => total | .count => count | .z => z

theorem eval_ordered (ops : Ops α) (env : Reg → α) (swap : Bool)
    (a b : Expr) :
    eval ops env (orderedExpr swap .add a b)=ordered swap ops.add (eval ops env a) (eval ops env b) := by
  cases swap <;> rfl

theorem local_mean (ops : Ops α) (code : Selected decoder N) (acc x : α) :
    eval ops (environment ops acc x ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero)
      (decoder.inspect code.object).meanUpdate=ordered code.plan.meanSwap ops.add acc x := by
  rw [code.recipes.meanNode]
  change _=ordered code.recipes.meanSwap ops.add acc x
  cases code.recipes.meanSwap <;> rfl

theorem local_carry (ops : Ops α) (code : Selected decoder N) (acc z : α) :
    eval ops (environment ops acc ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero z)
      (decoder.inspect code.object).varianceUpdate=carry ops code.plan.varianceCarry z acc := by
  rw [code.recipes.carryNode]
  change _=carry ops code.recipes.varianceCarry z acc
  cases h : code.recipes.varianceCarry with
  | fused => rfl
  | separate swap => cases swap <;> rfl

theorem local_divide (ops : Ops α) (code : Selected decoder N) (total : α) :
    eval ops (environment ops ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero total (ops.count N) ops.zero)
      (decoder.inspect code.object).meanDivide=divideCount ops code.plan total := by
  rw [code.recipes.meanDivision]
  unfold divideCount
  change _=(if code.recipes.eraseUnitDivision then total else ops.div total (ops.count N))
  cases code.recipes.eraseUnitDivision <;> rfl

theorem local_variance_divide (ops : Ops α) (code : Selected decoder N) (total : α) :
    eval ops (environment ops ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero total (ops.count N) ops.zero)
      (decoder.inspect code.object).varianceDivide=divideCount ops code.plan total := by
  rw [code.recipes.varianceDivision]
  unfold divideCount
  change _=(if code.recipes.eraseUnitDivision then total else ops.div total (ops.count N))
  cases code.recipes.eraseUnitDivision <;> rfl

theorem local_inverse (ops : Ops α) (code : Selected decoder N) (eps v : α) :
    eval ops (environment ops ops.zero ops.zero ops.zero ops.zero ops.zero eps v ops.zero ops.zero)
      (decoder.inspect code.object).inverse=inverseStd ops code.plan eps v := by
  rw [code.recipes.inverseNode]
  unfold inverseStd
  change _=ops.div ops.one (ops.sqrt (ordered code.recipes.epsSwap ops.add v eps))
  cases code.recipes.epsSwap <;> rfl

theorem local_output (ops : Ops α) (code : Selected decoder N) (mu inv x w : α) :
    eval ops (environment ops ops.zero x mu inv w ops.zero ops.zero ops.zero ops.zero)
      (decoder.inspect code.object).output=Native.output ops code.plan mu inv x w := by
  rw [code.recipes.outputNode]
  unfold Native.output cast32
  change _=ops.mul (ordered code.recipes.normalSwap ops.mul (ops.sub x mu) inv) w
  cases code.recipes.normalSwap <;> rfl

/-- The scalar recipe evaluator uses the decoded nodes of the selected bytes.
Physical execution composes these local equalities; the compiler adapter does
not assert any whole output or whole execution result. -/
def compiledMean (ops : Ops α) (code : Selected decoder N) (x : Nat → α) : α :=
  let acc := foldRange (fun off acc lane =>
    eval ops (environment ops (acc lane) (maskedX ops x N (off+lane.val))
      ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero)
      (decoder.inspect code.object).meanUpdate) 0 N 128 (fun _ : Fin 128 => ops.zero)
  eval ops (environment ops ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero
    ((decoder.inspect code.object).meanTree.eval ops.add acc) (ops.count N) ops.zero)
    (decoder.inspect code.object).meanDivide

def compiledVariance (ops : Ops α) (code : Selected decoder N) (x : Nat → α) (mu : α) : α :=
  let acc := foldRange (fun off acc lane =>
    let z := if off+lane.val<N then eval ops
      (environment ops ops.zero (maskedX ops x N (off+lane.val)) mu
        ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero) (decoder.inspect code.object).center
      else ops.zero
    eval ops (environment ops (acc lane) ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero z)
      (decoder.inspect code.object).varianceUpdate) 0 N 128 (fun _ : Fin 128 => ops.zero)
  eval ops (environment ops ops.zero ops.zero ops.zero ops.zero ops.zero ops.zero
    ((decoder.inspect code.object).varianceTree.eval ops.add acc) (ops.count N) ops.zero)
    (decoder.inspect code.object).varianceDivide

theorem compiled_mean (ops : Ops α) (code : Selected decoder N) (x : Nat → α) :
    compiledMean ops code x=Native.mean ops code.plan x := by
  unfold compiledMean
  simp_rw [local_mean]
  exact local_divide ops code _

theorem compiled_variance (ops : Ops α) (code : Selected decoder N) (x : Nat → α) (mu : α) :
    compiledVariance ops code x mu=Native.variance ops code.plan x mu := by
  unfold compiledVariance
  simp_rw [code.recipes.centerNode]
  simp only [centerExpr,eval,environment]
  simp_rw [local_carry]
  exact local_variance_divide ops code _

def execute (ops : Ops α) (code : Selected decoder shape.columns)
    (api : API α Store Byte) (layout : Layout shape strides) (eps : α)
    (rows : List (Row shape)) (m : Store) : Store :=
  rows.foldl (fun state row =>
    let x := xSnapshot ops api layout state row
    let mu := compiledMean ops code x
    let v := compiledVariance ops code x mu
    let inv := eval ops (environment ops ops.zero ops.zero ops.zero ops.zero ops.zero eps v ops.zero ops.zero)
      (decoder.inspect code.object).inverse
    (columns shape.columns 0).foldl (fun current col => api.write current (yAddress layout row col)
      (eval ops (environment ops ops.zero (api.read current (xAddress layout row col)) mu inv
        (api.read current (wAddress layout row col)) ops.zero ops.zero ops.zero ops.zero)
        (decoder.inspect code.object).output)) state) m

theorem execute_grid (ops : Ops α) (code : Selected decoder shape.columns)
    (api : API α Store Byte) (layout : Layout shape strides) (eps : α)
    (rows : List (Row shape)) (m : Store) :
    execute ops code api layout eps rows m=executeGrid ops code.plan api layout eps rows m := by
  unfold execute
  simp_rw [compiled_mean,compiled_variance,local_inverse,local_output]
  rfl

#axiomsClean local_mean
#axiomsClean local_carry
#axiomsClean compiled_mean
#axiomsClean compiled_variance
#axiomsClean execute_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end Compiler
end VeriTile.Bench.Repairs.LayernormFwdTriton
