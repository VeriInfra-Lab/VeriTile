import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Settings
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Model
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Compiler
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Runtime

/-! Exact source/object/ABI/options binding with local instruction recipes.
The decoder consumes the selected object bytes. Reduction permutations,
contractions and operand swaps are structural data, never numerical equations.
Offline SM80 profiles are bounded evidence, not this interface's full domain. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Compiler
open VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole
namespace Shared
abbrev ScalarABI := VeriTile.Bench.Repairs.LayernormFwdTriton.Compiler.ScalarABI
abbrev ScalarBinding := VeriTile.Bench.Repairs.LayernormFwdTriton.Compiler.ScalarBinding
abbrev HostScalar := VeriTile.Bench.Repairs.LayernormFwdTriton.Runtime.HostScalar
abbrev integerBinding := VeriTile.Bench.Repairs.LayernormFwdTriton.Runtime.integerBinding
end Shared
noncomputable section

def sourceSHA256 := "ca518babd0de189bce565bce2c0456ae94883c32cd5b844f0044df994bd4a636"
def originalSHA256 := "88893ebaf54a50b2b6d06841628e5a00a27f8200b026740ed6ad536deb83f6ef"

def warps (block : Nat) : Nat :=
  if 32768≤block then 32 else if 8192≤block then 16 else if 2048≤block then 8 else 4

def settings (N : Nat) : Except Nat (Nat × Nat) :=
  let block := N.nextPowerOfTwo
  if 65536<block then .error N else .ok (block,warps block)

theorem nextPower_covers (N : Nat) : N≤N.nextPowerOfTwo := Settings.nextPower_covers N

theorem settings_success (N block nw : Nat) (h : settings N=.ok (block,nw)) :
    block=N.nextPowerOfTwo ∧ N≤block ∧ block≤65536 ∧ nw=warps block := by
  dsimp only [settings] at h
  split at h
  · contradiction
  · cases h
    exact ⟨rfl,nextPower_covers N,by omega,rfl⟩

inductive Entry where
  | forward | backward | affineBackward
  deriving DecidableEq

structure Options where
  (target defaults : String)
  (numWarps numStages numCTAs : Nat)
  enableFusion : Bool

structure PrimitiveNames where
  (zero countConvert add sub mul div neg fma rsqrt epsilonABI : String)

structure Signature where
  pointers : List Nat
  scalars : List Shared.ScalarBinding
  epsilon : Option Shared.ScalarBinding
  hasBias : Bool
  block : Nat

structure Object where
  identity : Nat
  bytes : ByteArray
  sourceHash : String
  entry : Entry
  signature : Signature
  options : Options
  primitives : PrimitiveNames

inductive Reg where
  | x | mean | inv | dy | weight | bias | acc | first | second | eps | variance
  deriving DecidableEq
inductive Expr where
  | reg (r : Reg)
  | zero
  | add (a b : Expr) | sub (a b : Expr) | mul (a b : Expr)
  | neg (a : Expr) | fma (a b c : Expr) | rsqrt (a : Expr)
  deriving DecidableEq

def eval (ops : Ops α) (env : Reg → α) : Expr → α
  | .reg r => env r
  | .zero => ops.zero
  | .add a b => ops.add (eval ops env a) (eval ops env b)
  | .sub a b => ops.sub (eval ops env a) (eval ops env b)
  | .mul a b => ops.mul (eval ops env a) (eval ops env b)
  | .neg a => ops.neg (eval ops env a)
  | .fma a b c => ops.fma (eval ops env a) (eval ops env b) (eval ops env c)
  | .rsqrt a => ops.rsqrt (eval ops env a)

def orderedExpr (kind : Expr → Expr → Expr) (swap : Bool) (a b : Expr) :=
  if swap then kind b a else kind a b

def mulAddExpr : MulAdd → Expr → Expr → Expr → Expr
  | .separate sm sa,a,b,c => orderedExpr .add sa (orderedExpr .mul sm a b) c
  | .fused sm,a,b,c => if sm then .fma b a c else .fma a b c

def mulSubExpr : MulSub → Expr → Expr → Expr → Expr
  | .separate sm,a,b,c => .sub (orderedExpr .mul sm a b) c
  | .fused sm,a,b,c => if sm then .fma b a (.neg c) else .fma a b (.neg c)

def normedExpr (swap : Bool) := orderedExpr .mul swap (.sub (.reg .x) (.reg .mean)) (.reg .inv)

/-- Pointer-offset instruction recipes in source argument order. Each recipe
keeps the row/column arithmetic and the inactive affine column selection. -/
inductive AddressRecipe where
  | rowColumn (strideSlot : Nat) | row | column | maskedRowColumn | affineColumn
  deriving DecidableEq

def addressRecipes : Entry → List AddressRecipe
  | .forward => [.rowColumn 0,.rowColumn 1,.column,.column,.row,.row]
  | .backward => [.rowColumn 0,.rowColumn 1,.rowColumn 2,.column,.row,.row]
  | .affineBackward => [.affineColumn,.affineColumn,.maskedRowColumn,.maskedRowColumn,.row,.row]

def addressOffset (recipe : AddressRecipe) (arguments : List Nat) (row column N : Nat) : Nat :=
  match recipe with
  | .rowColumn slot => row*(arguments[slot]?.getD 0)+column
  | .row => row
  | .column => column
  | .maskedRowColumn => row*N+(if column<N then column else 0)
  | .affineColumn => column

structure Graph (B : Nat) where
  addresses : List AddressRecipe
  meanTree : Option (Nat → SumTree B)
  squareTree : Option (Nat → SquareTree B)
  firstTree : Option (Nat → ProductTree B)
  secondTree : Option (Nat → SumTree B)
  normal : Expr
  (output affine firstSubtract : Option Expr)
  (eraseInactiveAffine productSwap correctionSwap finalSwap epsilonSwap unitVarianceFMA unitCorrectionFMA eraseUnitDivision : Option Bool)
  (rowAxis columnIntegerWidth strideIntegerWidth rowIntegerWidth rowLoopStart rowLoopStep rowLoopBoundSlot : Nat)
  (centerMasked productMasked affineInactiveColumnZero fp32Loads fp32Stats fp32Arithmetic fp32Stores : Bool)

structure Decoder where
  decode : (B : Nat) → ByteArray → Signature → Options → PrimitiveNames → Graph B

def Decoder.inspect (decoder : Decoder) (B : Nat) (object : Object) :=
  decoder.decode B object.bytes object.signature object.options object.primitives

structure Selected (decoder : Decoder) (N B : Nat) where
  object : Object
  source : object.sourceHash=sourceSHA256
  block : object.signature.block=(if object.entry=.affineBackward then 128 else B)
  plan : Plan N B
  means : object.entry=.forward → (decoder.inspect B object).meanTree=some plan.meanTree
  squares : object.entry=.forward → (decoder.inspect B object).squareTree=some plan.squareTree
  firsts : object.entry=.backward → (decoder.inspect B object).firstTree=some plan.backwardFirst
  seconds : object.entry=.backward → (decoder.inspect B object).secondTree=some plan.backwardSecond
  normal : (decoder.inspect B object).normal=normedExpr plan.normedSwap
  output : object.entry=.forward → (decoder.inspect B object).output=some (mulAddExpr plan.outputFMA (.reg .x) (.reg .weight) (.reg .bias))
  affine : object.entry=.affineBackward → (decoder.inspect B object).affine=some (mulAddExpr plan.affineFMA (.reg .dy) (.reg .x) (.reg .acc))
  subtract : object.entry=.backward → (decoder.inspect B object).firstSubtract=some (mulSubExpr plan.firstSubtract (.reg .dy) (.reg .weight) (.reg .first))
  forwardTags : object.entry=.forward →
    (decoder.inspect B object).epsilonSwap=some plan.epsilonSwap ∧
    (decoder.inspect B object).unitVarianceFMA=some plan.unitVarianceFMA
  backwardTags : object.entry=.backward →
    (decoder.inspect B object).productSwap=some plan.productSwap ∧
    (decoder.inspect B object).correctionSwap=some plan.correctionSwap ∧
    (decoder.inspect B object).finalSwap=some plan.finalSwap ∧
    (decoder.inspect B object).unitCorrectionFMA=some plan.unitCorrectionFMA
  affineTags : object.entry=.affineBackward →
    (decoder.inspect B object).eraseInactiveAffine=some plan.eraseInactiveAffine
  divideTags : object.entry≠.affineBackward →
    (decoder.inspect B object).eraseUnitDivision=some plan.eraseUnitDivision
  addresses : (decoder.inspect B object).addresses=addressRecipes object.entry
  rowAxis : (decoder.inspect B object).rowAxis=0
  integers : (decoder.inspect B object).columnIntegerWidth=64 ∧ (decoder.inspect B object).strideIntegerWidth=64 ∧
    (decoder.inspect B object).rowIntegerWidth=64
  loop : object.entry=.affineBackward → (decoder.inspect B object).rowLoopStart=0 ∧ (decoder.inspect B object).rowLoopStep=1 ∧
    (decoder.inspect B object).rowLoopBoundSlot=0
  centerMask : object.entry=.forward → (decoder.inspect B object).centerMasked=Bool.true
  productMask : object.entry=.backward → (decoder.inspect B object).productMasked=Bool.true
  affineMask : object.entry=.affineBackward → (decoder.inspect B object).affineInactiveColumnZero=Bool.true
  precision : (decoder.inspect B object).fp32Loads=Bool.true ∧ (decoder.inspect B object).fp32Stats=Bool.true ∧
    (decoder.inspect B object).fp32Arithmetic=Bool.true ∧ (decoder.inspect B object).fp32Stores=Bool.true

/-- Each obligation is a scalar operation with its exact selected primitive.
No output, execution, memory, allocation or scheduling equation occurs here. -/
structure LocalConformance (ops : Ops UInt32) (names : PrimitiveNames)
    (targetZero : String → UInt32) (targetCount : String → Nat → UInt32)
    (targetBinary : String → UInt32 → UInt32 → UInt32)
    (targetUnary : String → UInt32 → UInt32)
    (targetFMA : String → UInt32 → UInt32 → UInt32 → UInt32) : Prop where
  zero : ops.zero=targetZero names.zero
  count : ∀ n,ops.count n=targetCount names.countConvert n
  add : ∀ a b,ops.add a b=targetBinary names.add a b
  sub : ∀ a b,ops.sub a b=targetBinary names.sub a b
  mul : ∀ a b,ops.mul a b=targetBinary names.mul a b
  div : ∀ a b,ops.div a b=targetBinary names.div a b
  neg : ∀ a,ops.neg a=targetUnary names.neg a
  fma : ∀ a b c,ops.fma a b c=targetFMA names.fma a b c
  rsqrt : ∀ a,ops.rsqrt a=targetUnary names.rsqrt a

structure Platform where
  decoder : Decoder
  scalar : PrimitiveNames → Ops UInt32
  epsilon : Object → Shared.HostScalar → UInt32
  epsilonBinding : Shared.ScalarBinding → Shared.HostScalar → Prop
  defaults : Entry → Nat → Options
  sourceWarps : ∀ entry block,(defaults entry block).numWarps=(if entry=.affineBackward then 4 else warps block)

structure Bound (platform : Platform) (code : Selected platform.decoder N B)
    (entry : Entry) (arguments : List Nat) (eps : Shared.HostScalar) (hasBias : Bool) : Prop where
  entryMatch : code.object.entry=entry
  scalars : List.Forall₂ Shared.integerBinding code.object.signature.scalars arguments
  epsilon : entry=.affineBackward ∧ code.object.signature.epsilon=none ∨
    ∃ binding,code.object.signature.epsilon=some binding ∧ platform.epsilonBinding binding eps
  bias : code.object.signature.hasBias=hasBias
  options : code.object.options=platform.defaults entry B

theorem eval_ordered (ops : Ops α) (env : Reg → α) (swap : Bool) (a b : Expr) :
    eval ops env (orderedExpr .mul swap a b)=ordered ops.mul swap (eval ops env a) (eval ops env b) := by
  cases swap <;> rfl

theorem eval_mulAdd (ops : Ops α) (env : Reg → α) (kind : MulAdd) (a b c : Expr) :
    eval ops env (mulAddExpr kind a b c)=mulAdd ops kind (eval ops env a) (eval ops env b) (eval ops env c) := by
  cases kind with
  | separate sm sa => cases sm <;> cases sa <;> rfl
  | fused sm => cases sm <;> rfl

theorem eval_mulSub (ops : Ops α) (env : Reg → α) (kind : MulSub) (a b c : Expr) :
    eval ops env (mulSubExpr kind a b c)=mulSub ops kind (eval ops env a) (eval ops env b) (eval ops env c) := by
  cases kind with
  | separate sm => cases sm <;> rfl
  | fused sm => cases sm <;> rfl

theorem selected_normal (code : Selected decoder N B) (ops : Ops α) (env : Reg → α) :
    eval ops env (decoder.inspect B code.object).normal=
      ordered ops.mul code.plan.normedSwap (ops.sub (env .x) (env .mean)) (env .inv) := by
  rw [code.normal]
  exact eval_ordered ops env code.plan.normedSwap _ _

theorem selected_output (code : Selected decoder N B) (entry : code.object.entry=.forward)
    (ops : Ops α) (env : Reg → α) :
    ((decoder.inspect B code.object).output.map (eval ops env))=
      some (mulAdd ops code.plan.outputFMA (env .x) (env .weight) (env .bias)) := by
  rw [code.output entry]
  simp only [Option.map_some,eval_mulAdd,eval]

theorem selected_affine (code : Selected decoder N B) (entry : code.object.entry=.affineBackward)
    (ops : Ops α) (env : Reg → α) :
    ((decoder.inspect B code.object).affine.map (eval ops env))=
      some (mulAdd ops code.plan.affineFMA (env .dy) (env .x) (env .acc)) := by
  rw [code.affine entry]
  simp only [Option.map_some,eval_mulAdd,eval]

theorem selected_subtract (code : Selected decoder N B) (entry : code.object.entry=.backward)
    (ops : Ops α) (env : Reg → α) :
    ((decoder.inspect B code.object).firstSubtract.map (eval ops env))=
      some (mulSub ops code.plan.firstSubtract (env .dy) (env .weight) (env .first)) := by
  rw [code.subtract entry]
  simp only [Option.map_some,eval_mulSub,eval]

theorem selected_warps (platform : Platform) (code : Selected platform.decoder N B)
    (bound : Bound platform code entry arguments eps hasBias) :
    code.object.options.numWarps=(if entry=.affineBackward then 4 else warps B) := by
  rw [bound.options]
  exact platform.sourceWarps entry B

#axiomsClean selected_warps
#axiomsClean selected_normal
#axiomsClean selected_output
#axiomsClean selected_affine
#axiomsClean selected_subtract
#axiomsClean nextPower_covers
#axiomsClean settings_success
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Compiler
