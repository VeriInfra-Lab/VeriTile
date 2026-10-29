import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Source
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.Geometry

/-! Ordinary compiler boundary: exact source/object/ABI/options plus local
address, control, cast, reduction-tree and scalar-expression recipes. This is
an interface for selected objects, not a theorem about Triton internals. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.Compiler
open VeriTile.Bench.Repairs.FastLayernorm (MulAdd)
namespace Shared
abbrev Binding := VeriTile.Bench.Repairs.FastLayernorm.Compiler.Shared.ScalarBinding
abbrev HostScalar := VeriTile.Bench.Repairs.FastLayernorm.Compiler.Shared.HostScalar
abbrev integerBinding := VeriTile.Bench.Repairs.FastLayernorm.Compiler.Shared.integerBinding
end Shared
noncomputable section

inductive Entry where | forward | groupedBackward | finalBackward
  deriving DecidableEq

def settings (N : Nat) : Except Nat (Nat × Nat) :=
  let B := min 16384 N.nextPowerOfTwo
  if N>B then .error N else .ok (B,Geometry.warps B)

theorem settings_success (N B W : Nat) (h : settings N=.ok (B,W)) :
    B=min 16384 N.nextPowerOfTwo ∧ N≤B ∧ B≤16384 ∧ W=Geometry.warps B := by
  dsimp only [settings] at h
  split at h
  · contradiction
  · cases h
    exact ⟨rfl,by omega,Nat.min_le_left _ _,rfl⟩

structure Options where
  (target defaults : String)
  (numWarps numStages numCTAs : Nat)
  fusion : Bool

structure Signature where
  pointerBits : List Nat
  scalars : List Shared.Binding
  epsilon : Option Shared.Binding
  block : Nat
  group : Nat
  rowBlock : Nat

structure Primitives where
  (zero one count add sub mul div fma sqrt epsilon : String)

structure Object where
  identity : Nat
  bytes : ByteArray
  sourceHash : String
  entry : Entry
  signature : Signature
  options : Options
  primitives : Primitives

inductive Reg where | x | mean | inverse | dy | w | bias | first | second | eps | variance | previous
  deriving DecidableEq
inductive Expr where
  | reg (r : Reg)
  | zero | one
  | add (a b : Expr) | sub (a b : Expr) | mul (a b : Expr)
  | div (a b : Expr) | fma (a b c : Expr) | sqrt (a : Expr)
  deriving DecidableEq

def eval (ops : Ops α) (env : Reg → α) : Expr → α
  | .reg r => env r
  | .zero => ops.zero | .one => ops.one
  | .add a b => ops.add (eval ops env a) (eval ops env b)
  | .sub a b => ops.sub (eval ops env a) (eval ops env b)
  | .mul a b => ops.mul (eval ops env a) (eval ops env b)
  | .div a b => ops.div (eval ops env a) (eval ops env b)
  | .fma a b c => ops.fma (eval ops env a) (eval ops env b) (eval ops env c)
  | .sqrt a => ops.sqrt (eval ops env a)

def orderedExpr (op : Expr → Expr → Expr) (swap : Bool) (a b : Expr) := if swap then op b a else op a b

def mulAddExpr : MulAdd → Expr → Expr → Expr → Expr
  | .separate sm sa,a,b,c => orderedExpr .add sa (orderedExpr .mul sm a b) c
  | .fused sm,a,b,c => if sm then .fma b a c else .fma a b c

def normal (p : Plan N B) := orderedExpr .mul p.normalSwap (.sub (.reg .x) (.reg .mean)) (.reg .inverse)
def inverse (p : Plan N B) := Expr.div .one (.sqrt (orderedExpr .add p.epsilonSwap (.reg .variance) (.reg .eps)))
def output (p : Plan N B) := mulAddExpr p.outputKind (normal p) (.reg .w) (.reg .bias)
def weighted (p : Plan N B) := orderedExpr .mul p.weightDySwap (.reg .w) (.reg .dy)
def dx (p : Plan N B) := orderedExpr .mul p.finalSwap
  (.sub (weighted p) (mulAddExpr p.correctionKind (normal p) (.reg .first) (.reg .second))) (.reg .inverse)
def firstDW (p : Plan N B) := orderedExpr .mul p.groupFirstSwap (.reg .dy) (normal p)
def laterDW (p : Plan N B) := mulAddExpr p.groupKind (.reg .dy) (normal p) (.reg .previous)
def laterDB (p : Plan N B) := orderedExpr .add p.dbSwap (.reg .dy) (.reg .previous)

inductive Address where | rowColumn | column | row | groupColumn | tileRowColumn
  deriving DecidableEq

def addresses : Entry → List Address
  | .forward => [.rowColumn,.rowColumn,.column,.column,.row,.row]
  | .groupedBackward => [.rowColumn,.rowColumn,.groupColumn,.groupColumn,.rowColumn,.column,.row,.row]
  | .finalBackward => [.tileRowColumn,.tileRowColumn,.column,.column]

inductive Control where
  | oneRow | groupedRows | tiledGroups
  deriving DecidableEq

structure Recipe (N B : Nat) where
  addresses : List Address
  control : Control
  integerWidth : Nat
  fp32Loads : Bool
  fp32Carries : Bool
  fp32Stores : Bool
  meanTree : Nat → SumTree B
  varianceTree : Nat → ProductTree B
  firstTree : Nat → ProductTree B
  secondTree : Nat → SumTree B
  finalTree : Nat → SumTree 32
  (normal inverse output dx firstDW laterDW laterDB : Expr)
  (centerMask reductionMask firstWithoutZeroAdd preservePartialCast eraseUnitDivision finalStreamSwap : Bool)

structure Decoder where
  decode : (N B : Nat) → ByteArray → Signature → Options → Primitives → Option (Recipe N B)

structure Selected (decoder : Decoder) (entry : Entry) (N B : Nat) where
  object : Object
  source : object.sourceHash=Source.candidateSHA256
  entryEq : object.entry=entry
  plan : Plan N B
  finalPlan : FinalPlan
  recipe : Recipe N B
  decoded : decoder.decode N B object.bytes object.signature object.options object.primitives=some recipe
  addressMatch : recipe.addresses=addresses entry
  controlMatch : recipe.control=(match entry with | .forward => .oneRow | .groupedBackward => .groupedRows | .finalBackward => .tiledGroups)
  precision : recipe.integerWidth=64 ∧ recipe.fp32Loads=Bool.true ∧ recipe.fp32Carries=Bool.true ∧ recipe.fp32Stores=Bool.true
  means : entry=.forward → recipe.meanTree=plan.meanTree
  variances : entry=.forward → recipe.varianceTree=plan.varianceTree
  firsts : entry=.groupedBackward → recipe.firstTree=plan.firstTree
  seconds : entry=.groupedBackward → recipe.secondTree=plan.secondTree
  finals : entry=.finalBackward → recipe.finalTree=finalPlan.tree
  normalMatch : entry≠.finalBackward → recipe.normal=normal plan
  inverseMatch : entry=.forward → recipe.inverse=inverse plan
  outputMatch : entry=.forward → recipe.output=output plan
  dxMatch : entry=.groupedBackward → recipe.dx=dx plan
  firstDWMatch : entry=.groupedBackward → recipe.firstDW=firstDW plan
  laterDWMatch : entry=.groupedBackward → recipe.laterDW=laterDW plan
  laterDBMatch : entry=.groupedBackward → recipe.laterDB=laterDB plan
  forwardMasks : entry=.forward → recipe.centerMask=Bool.true
  backwardMasks : entry=.groupedBackward → recipe.reductionMask=Bool.true ∧ recipe.firstWithoutZeroAdd=Bool.true ∧ recipe.preservePartialCast=Bool.true
  countDivision : entry≠.finalBackward → recipe.eraseUnitDivision=plan.eraseUnitDivision
  streamOrder : entry=.finalBackward → recipe.finalStreamSwap=finalPlan.streamSwap

structure Platform where
  decoder : Decoder
  scalar : Primitives → Ops UInt32
  epsilon : Object → Shared.HostScalar → UInt32
  epsilonBinding : Shared.Binding → Shared.HostScalar → Prop
  defaults : Entry → Nat → Options
  warps : ∀ e B,(defaults e B).numWarps=(if e=.finalBackward then 4 else Geometry.warps B)

structure Bound (platform : Platform) (code : Selected platform.decoder entry N B)
    (M : Nat) (eps : Shared.HostScalar) : Prop where
  scalarArguments : List.Forall₂ Shared.integerBinding code.object.signature.scalars
    (match entry with | .forward => [N,N] | .groupedBackward => [N,M,N] | .finalBackward => [min (Geometry.groups N) M,N])
  pointerTypes : code.object.signature.pointerBits=List.replicate (if entry=.groupedBackward then 8 else if entry=.forward then 6 else 4) 32
  blocks : code.object.signature.block=(if entry=.finalBackward then 128 else B)
  groups : entry=.groupedBackward → code.object.signature.group=Geometry.groups N
  rowBlock : entry=.finalBackward → code.object.signature.rowBlock=32
  epsilonABI : entry=.forward → ∃ b,code.object.signature.epsilon=some b ∧ platform.epsilonBinding b eps
  options : code.object.options=platform.defaults entry B

theorem eval_ordered (ops : Ops α) (env : Reg → α) (swap : Bool) (a b : Expr) :
    eval ops env (orderedExpr .mul swap a b)=ordered ops.mul swap (eval ops env a) (eval ops env b) := by
  cases swap <;> rfl

theorem eval_mulAdd (ops : Ops α) (env : Reg → α) (kind : MulAdd) (a b c : Expr) :
    eval ops env (mulAddExpr kind a b c)=mulAdd ops kind (eval ops env a) (eval ops env b) (eval ops env c) := by
  cases kind with
  | separate sm sa => cases sm <;> cases sa <;> rfl
  | fused sm => cases sm <;> rfl

/-- Numerical target-bit obligations are only operationwise equalities on all
32-bit words; they do not contain any layout, memory or successful-run clause. -/
abbrev LocalConformance := @LocalExact UInt32

#axiomsClean settings_success
#axiomsClean eval_mulAdd
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.Compiler
