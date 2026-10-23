import bench.repairs.tritonbench_g.triton_argmax.revision004.Proofs.NativeTransport

/-! Finite typed SSA derivations with a total, constructed interpreter.
Every node names its actual raw instruction and operand references. The type
map records native index widths. Index literals carry local representability
certificates; selects and transports preserve width, and sign extension has
an explicit source/destination-width check. There is no arbitrary-graph
Evaluation function. A decoded body will contain finite derivations for its
entire evaluated graph, including otherwise unused nodes.
-/
namespace VeriTile.Bench.Repairs.TritonArgmax.Revision004.ScalarIR
open Source Reduction SelectedEmission

inductive Kind where
  | value
  | index : Nat → Kind
  | raw : Nat → Kind
  | predicate
  | effect
  deriving DecidableEq

/-- Native payload width, independent of the interpretation of floating values. -/
def Kind.bits (valueBits : Nat) : Kind → Nat
  | .value => valueBits
  | .raw width => width
  | .index width => width
  | .predicate => 1
  | .effect => 0

abbrev Meaning (α : Type) : Kind → Type
  | .value => α
  | .index width => BitVec width
  | .predicate => Bool
  | .effect => Unit
  | .raw _ => Unit

inductive Expr {n : Nat} (valueBits : Nat) (graph : Graph n) (types : Ref → Kind) : Kind → Ref → Type where
  | input (site : Ref) (lane : Fin n)
      (typed : types site=.value) (instruction : graph.instruction site=some (.input lane)) :
      Expr valueBits graph types .value site
  | index (site : Ref) (width : Nat) (positive : 0<width) (lane : Fin n)
      (fits : lane.val<2^(width-1)) (typed : types site=.index width)
      (instruction : graph.instruction site=some (.laneIndex lane)) : Expr valueBits graph types (.index width) site
  | greater (site : Ref) (left : Expr valueBits graph types .value a) (right : Expr valueBits graph types .value b)
      (typed : types site=.predicate) (instruction : graph.instruction site=some (.greater a b)) :
      Expr valueBits graph types .predicate site
  | equal (site : Ref) (left : Expr valueBits graph types .value a) (right : Expr valueBits graph types .value b)
      (typed : types site=.predicate) (instruction : graph.instruction site=some (.equal a b)) :
      Expr valueBits graph types .predicate site
  | less (site : Ref) (left : Expr valueBits graph types (.index width) a) (right : Expr valueBits graph types (.index width) b)
      (typed : types site=.predicate) (instruction : graph.instruction site=some (.indexLess a b)) :
      Expr valueBits graph types .predicate site
  | both (site : Ref) (left : Expr valueBits graph types .predicate a) (right : Expr valueBits graph types .predicate b)
      (typed : types site=.predicate) (instruction : graph.instruction site=some (.boolAnd a b)) :
      Expr valueBits graph types .predicate site
  | either (site : Ref) (left : Expr valueBits graph types .predicate a) (right : Expr valueBits graph types .predicate b)
      (typed : types site=.predicate) (instruction : graph.instruction site=some (.boolOr a b)) :
      Expr valueBits graph types .predicate site
  | selectValue (site : Ref) (condition : Expr valueBits graph types .predicate p)
      (yes : Expr valueBits graph types .value a) (no : Expr valueBits graph types .value b)
      (typed : types site=.value) (instruction : graph.instruction site=some (.selectValue p a b)) :
      Expr valueBits graph types .value site
  | selectIndex (site : Ref) (condition : Expr valueBits graph types .predicate p)
      (yes : Expr valueBits graph types (.index width) a) (no : Expr valueBits graph types (.index width) b)
      (typed : types site=.index width) (instruction : graph.instruction site=some (.selectIndex p a b)) :
      Expr valueBits graph types (.index width) site
  | transport (site : Ref) (source : Expr valueBits graph types kind sourceSite)
      (typed : types site=kind) (edge : NativeTransport.Checked graph (kind.bits valueBits) site sourceSite) : Expr valueBits graph types kind site
  | splitLow (site : Ref) (source : Expr valueBits graph types .value sourceSite)
      (wordWidth : valueBits=64) (typed : types site=.raw 32)
      (instruction : graph.instruction site=some (.splitLow sourceSite)) : Expr valueBits graph types (.raw 32) site
  | splitHigh (site : Ref) (source : Expr valueBits graph types .value sourceSite)
      (wordWidth : valueBits=64) (typed : types site=.raw 32)
      (instruction : graph.instruction site=some (.splitHigh sourceSite)) : Expr valueBits graph types (.raw 32) site
  | rawExtend (site : Ref) (source : Expr valueBits graph types (.raw 32) sourceSite)
      (typed : types site=.raw 64) (instruction : graph.instruction site=some (.zeroExtend32 sourceSite)) :
      Expr valueBits graph types (.raw 64) site
  | rawShift (site : Ref) (source : Expr valueBits graph types (.raw 64) sourceSite)
      (typed : types site=.raw 64) (instruction : graph.instruction site=some (.shiftLeft32 sourceSite)) :
      Expr valueBits graph types (.raw 64) site
  | widen (site : Ref) (source : Expr valueBits graph types (.index width) sourceSite) (larger : width≤target)
      (typed : types site=.index target) (instruction : graph.instruction site=some (.copy sourceSite))
      (earlier : graph.time sourceSite<graph.time site) : Expr valueBits graph types (.index target) site
  | sharedStore (site : Ref) (slot width : Nat) (guard : LaneGuard)
      (source : Expr valueBits graph types kind sourceSite) (completeWord : width*8=(kind.bits valueBits)) (typed : types site=.effect)
      (instruction : graph.instruction site=some (.sharedStore slot width sourceSite guard)) :
      Expr valueBits graph types .effect site
  | barrier (site : Ref) (identity : Nat) (typed : types site=.effect)
      (instruction : graph.instruction site=some (.barrier identity)) : Expr valueBits graph types .effect site

def Expr.eval {n : Nat} {graph : Graph n} {types : Ref → Kind} {kind : Kind} {site : Ref}
    (gt eq : α → α → Bool) (input : Fin n → α) : Expr valueBits graph types kind site → Meaning α kind
  | .input _ lane _ _ => input lane
  | .index _ width _ lane _ _ _ => BitVec.ofNat width lane.val
  | .greater _ left right _ _ => gt (left.eval gt eq input) (right.eval gt eq input)
  | .equal _ left right _ _ => eq (left.eval gt eq input) (right.eval gt eq input)
  | .less _ left right _ _ => decide ((left.eval gt eq input).toInt<(right.eval gt eq input).toInt)
  | .both _ left right _ _ => left.eval gt eq input && right.eval gt eq input
  | .either _ left right _ _ => left.eval gt eq input || right.eval gt eq input
  | .selectValue _ condition yes no _ _ => if condition.eval gt eq input=Bool.true then yes.eval gt eq input else no.eval gt eq input
  | .selectIndex _ condition yes no _ _ => if condition.eval gt eq input=Bool.true then yes.eval gt eq input else no.eval gt eq input
  | .transport _ source _ _ => source.eval gt eq input
  | .splitLow _ _ _ _ _ => ()
  | .splitHigh _ _ _ _ _ => ()
  | .rawExtend _ _ _ _ => ()
  | .rawShift _ _ _ _ => ()
  | .widen _ source _ _ _ _ => (source.eval gt eq input).signExtend _
  | .sharedStore _ _ _ _ _ _ _ _ => ()
  | .barrier _ _ _ _ => ()

/-- Native values are complete bit vectors at the program's actual value
width. Raw half-register expressions have their own 32/64-bit carriers. -/
abbrev NativeMeaning (valueBits : Nat) : Kind → Type
  | .value => BitVec valueBits
  | .index width => BitVec width
  | .raw width => BitVec width
  | .predicate => Bool
  | .effect => Unit

def eraseNative : (kind : Kind) → NativeMeaning valueBits kind → Meaning (BitVec valueBits) kind
  | .value,x => x
  | .index _,x => x
  | .predicate,x => x
  | .effect,x => x
  | .raw _,_ => ()

def Expr.nativeEval {n : Nat} {graph : Graph n} {types : Ref → Kind} {kind : Kind} {site : Ref}
    (gt eq : BitVec valueBits → BitVec valueBits → Bool) (input : Fin n → BitVec valueBits) :
    Expr valueBits graph types kind site → NativeMeaning valueBits kind
  | .input _ lane _ _ => input lane
  | .index _ width _ lane _ _ _ => BitVec.ofNat width lane.val
  | .greater _ left right _ _ => gt (left.nativeEval gt eq input) (right.nativeEval gt eq input)
  | .equal _ left right _ _ => eq (left.nativeEval gt eq input) (right.nativeEval gt eq input)
  | .less _ left right _ _ => decide ((left.nativeEval gt eq input).toInt<(right.nativeEval gt eq input).toInt)
  | .both _ left right _ _ => left.nativeEval gt eq input && right.nativeEval gt eq input
  | .either _ left right _ _ => left.nativeEval gt eq input || right.nativeEval gt eq input
  | .selectValue _ condition yes no _ _ => if condition.nativeEval gt eq input=Bool.true then yes.nativeEval gt eq input else no.nativeEval gt eq input
  | .selectIndex _ condition yes no _ _ => if condition.nativeEval gt eq input=Bool.true then yes.nativeEval gt eq input else no.nativeEval gt eq input
  | @Expr.transport _ _ _ _ kind _ _ source _ edge =>
    match kind with
    | .value => edge.transfer (source.nativeEval gt eq input)
    | .index _ => edge.transfer (source.nativeEval gt eq input)
    | .raw _ => edge.transfer (source.nativeEval gt eq input)
    | .predicate => source.nativeEval gt eq input
    | .effect => ()
  | .splitLow _ source width _ _ => NativePayload.low ((source.nativeEval gt eq input).cast width)
  | .splitHigh _ source width _ _ => NativePayload.high ((source.nativeEval gt eq input).cast width)
  | .rawExtend _ source _ _ => (source.nativeEval gt eq input).setWidth 64
  | .rawShift _ source _ _ => source.nativeEval gt eq input <<< 32
  | .widen _ source _ _ _ _ => (source.nativeEval gt eq input).signExtend _
  | .sharedStore _ _ _ _ _ _ _ _ => ()
  | .barrier _ _ _ _ => ()

/-- Exact native-to-erased refinement is derived for every finite expression.
No opaque value-preservation, whole-output or arbitrary evaluator law is used. -/
theorem Expr.native_erases (gt eq : BitVec valueBits → BitVec valueBits → Bool)
    (input : Fin n → BitVec valueBits) (expression : Expr valueBits graph types kind site) :
    eraseNative kind (expression.nativeEval gt eq input)=expression.eval gt eq input := by
  induction expression with
  | @transport kind sourceSite site source typed edge ih =>
    cases kind <;> simp_all [nativeEval,eval,eraseNative,NativeTransport.Checked.transfer_exact]
  | _ => simp_all [nativeEval,eval,eraseNative]

theorem index_literal_exact (width value : Nat) (positive : 0<width) (fits : value<2^(width-1)) :
    (BitVec.ofNat width value).toInt=Int.ofNat value := by
  have power : 2^width=2*2^(width-1) := by
    calc
      2^width=2^((width-1)+1) := congrArg (fun exponent => 2^exponent) (by omega)
      _=2*2^(width-1) := by rw [Nat.pow_succ,Nat.mul_comm]
  have small : value<2^width := by rw [power]; omega
  have bits : (BitVec.ofNat width value).toNat=value := by
    simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt small]
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [bits,power]; omega),bits]
  rfl

def Nonnegative {n : Nat} {graph : Graph n} {types : Ref → Kind} {kind : Kind} {site : Ref}
    (gt eq : α → α → Bool) (input : Fin n → α) (expression : Expr valueBits graph types kind site) : Prop :=
  match kind with
  | .index _ => 0≤(expression.eval gt eq input).toInt
  | _ => True

theorem index_nonnegative (gt eq : α → α → Bool) (input : Fin n → α)
    (expression : Expr valueBits graph types kind site) : Nonnegative gt eq input expression := by
  induction expression with
  | index site width positive lane fits typed instruction =>
    change 0≤(BitVec.ofNat width lane.val).toInt
    rw [index_literal_exact width lane.val positive fits]
    exact Int.natCast_nonneg _
  | selectIndex site condition yes no typed instruction ihCondition ihYes ihNo =>
    change 0≤(if condition.eval gt eq input=Bool.true then yes.eval gt eq input else no.eval gt eq input).toInt
    split <;> assumption
  | @transport kind sourceSite site source typed edge ih => cases kind <;> exact ih
  | widen site source larger typed instruction earlier ih =>
    change 0≤((source.eval gt eq input).signExtend _).toInt
    rw [BitVec.toInt_signExtend_of_le larger]
    exact ih
  | _ => trivial

theorem index_signed_natural (gt eq : α → α → Bool) (input : Fin n → α)
    (expression : Expr valueBits graph types (.index width) site) :
    (expression.eval gt eq input).toInt=Int.ofNat (expression.eval gt eq input).toNat := by
  have nonnegative := index_nonnegative gt eq input expression
  change 0≤(expression.eval gt eq input).toInt at nonnegative
  apply BitVec.toInt_eq_toNat_of_msb
  rw [BitVec.msb_eq_toInt]
  simp only [not_lt.mpr nonnegative,decide_false]

structure PackedExpr {n : Nat} (valueBits : Nat) (graph : Graph n) (types : Ref → Kind) where
  kind : Kind
  site : Ref
  expression : Expr valueBits graph types kind site

/-- Finite complete evaluated graph. Every actual instruction reference is
covered by a finite typed derivation. In particular a disconnected cyclic
component cannot be hidden behind an otherwise valid reduction root. -/
structure Code {n : Nat} (valueBits : Nat) (graph : Graph n) where
  types : Ref → Kind
  expressions : List (PackedExpr valueBits graph types)
  complete : ∀ site,graph.instruction site≠none →
    ∃ expression,expression∈expressions ∧ expression.site=site

def Code.evaluate (code : Code (n:=n) valueBits graph) (gt eq : α → α → Bool) (input : Fin n → α) :
    List ((kind : Kind) × Meaning α kind) :=
  code.expressions.map fun expression => ⟨expression.kind,expression.expression.eval gt eq input⟩

#axiomsClean Expr.native_erases
#axiomsClean index_literal_exact
#axiomsClean index_nonnegative
#axiomsClean index_signed_natural
end VeriTile.Bench.Repairs.TritonArgmax.Revision004.ScalarIR
