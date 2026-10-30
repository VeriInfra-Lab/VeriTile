import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.ContextAttnMistral.NativeGraph
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 2000000

/-- Local native instruction ports. There is no split-rounded dot primitive,
compiler-law callback or whole-program function. DotAcc retains actual input
formats/options, tile coordinates and a distinct accumulator operand. -/
structure Primitives (α : Type) where
  leaf : FloatDType → BitVec 32 → α
  literal : Option ComputeDType → FloatDType → ℝ → α
  fromNat : Option ComputeDType → Nat → α
  fromInt : Option ComputeDType → Int → α
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  unary : Option ComputeDType → Unary → α → α
  binary : Option ComputeDType → FloatDType → Binary → α → α → α
  fma : Option ComputeDType → FloatDType → α → α → α → α
  eq : Option ComputeDType → FloatDType → α → α → Bool
  dotAcc : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    (batch : TileShape) → (m k n : Nat) →
    (TileIndex (batch++[m,k]) → α) → (TileIndex (batch++[k,n]) → α) →
    α → TileIndex (batch++[m,n]) → α

/-- Evaluation happens after graph construction. A product consumed by FMA
has no intermediate rounded value. A standalone pending dot materializes as
one zero-accumulator dotAcc; the source's two immediate add consumers replace
that payload before evaluation, retaining their actual accumulator instead. -/
def evaluate {α : Type} (ops : Primitives α) : Expr → α
  | .leaf d w => ops.leaf d w
  | .literal p d x => ops.literal p d x
  | .fromNat p x => ops.fromNat p x
  | .fromInt p x => ops.fromInt p x
  | .cast p source target x => ops.cast p source target (evaluate ops x)
  | .unary p op x => ops.unary p op (evaluate ops x)
  | .binary p d op x y => ops.binary p d op (evaluate ops x) (evaluate ops y)
  | .product p d reverse x y =>
      if reverse then ops.binary p d .mul (evaluate ops y) (evaluate ops x)
      else ops.binary p d .mul (evaluate ops x) (evaluate ops y)
  | .fma p d x y acc => ops.fma p d (evaluate ops x) (evaluate ops y) (evaluate ops acc)
  | .pendingDot p l r out opts batch m k n a b i =>
      ops.dotAcc p l r out opts batch m k n
        (fun j => evaluate ops (a j)) (fun j => evaluate ops (b j)) (ops.literal p out 0) i
  | .dotAcc p l r out opts batch m k n a b acc i =>
      ops.dotAcc p l r out opts batch m k n
        (fun j => evaluate ops (a j)) (fun j => evaluate ops (b j)) (evaluate ops acc) i

def equality {α : Type} (ops : Primitives α) : Equality := fun p d a b =>
  ops.eq p d (evaluate ops a) (evaluate ops b)

theorem evaluate_consumed_dot {α : Type} (ops : Primitives α) (plan : Plan)
    (p dp : Option ComputeDType) (d l r out : FloatDType) (opts : DotOptions)
    (batch : TileShape) (m k n : Nat)
    (a : TileIndex (batch++[m,k]) → Expr) (b : TileIndex (batch++[k,n]) → Expr)
    (acc : Expr) (i : TileIndex (batch++[m,n])) :
    evaluate ops (consume plan p d .add acc (.pendingDot dp l r out opts batch m k n a b i)) =
      ops.dotAcc dp l r out opts batch m k n
        (fun j => evaluate ops (a j)) (fun j => evaluate ops (b j)) (evaluate ops acc) i := rfl

theorem evaluate_normalizer {α : Type} (ops : Primitives α) (plan : Plan)
    (p : Option ComputeDType) (d : FloatDType) (alpha li beta lij : Expr) :
    evaluate ops (consume plan p d .add
      (consume plan p d .mul alpha li) (consume plan p d .mul beta lij)) =
      let right := if plan.reverseMul p (nativeDType d) beta lij then
        ops.binary p (nativeDType d) .mul (evaluate ops lij) (evaluate ops beta)
        else ops.binary p (nativeDType d) .mul (evaluate ops beta) (evaluate ops lij)
      if plan.reverseFMA p (nativeDType d) alpha li then
        ops.fma p (nativeDType d) (evaluate ops li) (evaluate ops alpha) right
      else ops.fma p (nativeDType d) (evaluate ops alpha) (evaluate ops li) right := by
  rw [normalizer]
  split <;> rfl

theorem evaluate_exp {α : Type} (ops : Primitives α) (p : Option ComputeDType) (x : Expr) :
    evaluate ops (nativeUnary p .exp x) =
      ops.unary p .exp2 (ops.binary p .fp32 .mul (evaluate ops x) (ops.leaf .fp32 0x3fb8aa3b)) := rfl

/-- The selected FP32 entry uses actual input words without a numerical
callback that could modify them. Storage-format conversions remain explicit. -/
def rawWords (ops : Primitives (BitVec 32)) : Primitives (BitVec 32) :=
  { ops with leaf := fun _ word => word }

def logicalWords (query key value : Nat → Nat → Nat → BitVec 32) : FPHost.LogicalInputs Expr :=
  ⟨fun row head dim => .leaf .fp32 (query row head dim),
   fun row head dim => .leaf .fp32 (key row head dim),
   fun row head dim => .leaf .fp32 (value row head dim)⟩

theorem evaluate_leaf (ops : Primitives (BitVec 32)) (d : FloatDType) (word : BitVec 32) :
    evaluate (rawWords ops) (.leaf d word) = word := rfl

/-- Native numerical interpretation of the successful complete graph. The
reference is the same full logical source under staged instruction semantics;
its readback is evaluated using only the local native ports above. -/
theorem evaluated_whole_call {α : Type} (ops : Primitives α) (plan : Plan)
    (c : Host.Call) (storage : Storage) (s : State Expr)
    (logical : FPHost.LogicalInputs Expr) (view : OutputView.View FPHost.Coord Expr)
    (before : Nat → Expr) (prepared : FPHost.Prepared c storage s logical view before)
    (writable : view.Writable (FPHost.active c))
    (buffers : NativeIndex.Buffers) (native : NativeIndex.Ready c storage buffers) :
    ∃ result reference,
      FPHost.Exec (backend plan (equality ops)) c storage s view before result ∧
      FPHost.Composition.schedule (backend plan (equality ops)).algebra (FPHost.source c storage) c.grid
        (FPHost.Composition.allPrograms c.grid) s = some reference ∧
      result.1 = none ∧
      (∀ i, FPHost.active c i → evaluate ops (view.read result.2 i) =
        evaluate ops (FPHost.values c storage reference i)) ∧
      (∀ a, (¬∃ i, FPHost.active c i ∧ view.addr i = a) →
        evaluate ops (result.2 a) = evaluate ops (before a)) ∧
      (∀ rg, rg ∈ [c.regions.query,c.regions.key,c.regions.value,c.regions.start,c.regions.sequence] →
        ∀ off, reference.mem rg off = s.mem rg off) ∧
      NativeIndex.Correspondence c storage buffers := by
  obtain ⟨⟨result,hex,ref,href,hread,hframe,hinputs⟩,hnative⟩ :=
    whole_call plan (equality ops) c storage s logical view before prepared writable buffers native
  refine ⟨result,ref,hex,href,hex.1,?_,?_,hinputs,hnative⟩
  · intro i hi
    exact congrArg (evaluate ops) (hread i hi)
  · intro a ha
    exact congrArg (evaluate ops) (hframe a ha)

#axiomsClean evaluate_consumed_dot
#axiomsClean evaluate_normalizer
#axiomsClean evaluate_exp
#axiomsClean evaluated_whole_call
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ContextAttnMistral.NativeGraph
