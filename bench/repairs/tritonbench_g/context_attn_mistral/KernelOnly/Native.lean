import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativeProfiles
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.WindowTrees
import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativeInstructions

namespace VeriTile.Bench.Repairs.ContextAttnMistral.KernelNative
open VeriTile.Triton FP.Structural
open NativePhysical (Expr Owner Edge Plan evalTree)
open NativeGraph (OrderedTree)
open NativeInstructions (primitives dotAt)
open VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph (Word ScalarOps)
noncomputable section

/-- Only source graph shape and integer coordinates are retained. Raw words,
Real literals, comparisons, and numerical results are absent. -/
inductive Form where
  | atom | literal | product | reduction
  | unary (op : Unary)
  | binary (op : Binary)
  | dot (m k n : Nat) (productSeed : Bool) (linear : Nat)

def form : Expr → Form
  | .literal .. => .literal
  | .product .. => .product
  | .reduction .. => .reduction
  | .unary _ op _ => .unary op
  | .binary _ _ op _ _ => .binary op
  | .dotAcc _ _ _ _ _ batch m k n _ _ acc i =>
      .dot m k n (match acc with | .product .. => Bool.true | _ => Bool.false)
        (TileShape.linearIndex (batch++[m,n]) i)
  | _ => .atom

def reverseMul : Form → Form → Bool
  | .dot ..,.literal => Bool.true
  | .unary ..,.reduction => Bool.true
  | .binary .div,.unary .. => Bool.true
  | _,_ => Bool.false

def route (d : Nat) : Form → Edge → Owner → Owner
  | _,.reductionInput column,owner => owner/4*4+column%8/2
  | .dot 128 128 _ Bool.true _,.dotLeft linear,_ =>
      NativeProfiles.producer d (linear/128) (linear%128)
  | _,_,owner => owner

/-- Fixed SM90 source profile; no caller plan or route is accepted. -/
def plan (d window : Nat) : Plan where
  reverseMul := fun _ _ a b _ => reverseMul (form a) (form b)
  reverseFMA := fun _ _ _ _ _ => Bool.true
  sumTree := fun {shape} ax keep i owner => KernelWindowTrees.select d window
    (TileShape.linearIndex (TileShape.reduceShape shape ax keep) i) owner _ Bool.false
  maxTree := fun {shape} ax keep i owner => KernelWindowTrees.select d window
    (TileShape.linearIndex (TileShape.reduceShape shape ax keep) i) owner _ Bool.true
  route := fun e edge owner => route d (form e) edge owner

def owner (d : Nat) (e : Expr) : Owner :=
  match form e with
  | .dot _ _ n _ linear => NativeProfiles.producer d (linear/n) (linear%n)
  | _ => 0

/-- The only reached literal-plus-leaf addition is the constant initializer
zero plus negative infinity. The compiler folds that exact source site.
No leaf or literal payload is inspected to choose the rewrite. -/
def consume (p : Option ComputeDType) (d : FloatDType) : Binary → Expr → Expr → Expr
  | .add,.literal ..,value@(.leaf ..) => value
  | op,a,b => NativePhysical.consume p d op a b

def backend (d window : Nat) : FPBackend.Ops Expr :=
  { NativePhysical.backend (plan d window) with
    scalar := { NativePhysical.algebra (plan d window) with binary:=consume } }

/-- Cleared QK is a source literal initializer; every live PV seed is a
product constructor. Neither raw words nor literal values select scale-d. -/
def live : Expr → Bool
  | .literal .. => Bool.false
  | _ => Bool.true

theorem initializer_fold (d window : Nat) :
    (backend d window).scalar.binary (some .fp32) .real .add
      ((backend d window).scalar.literal (some .fp32) .real 0) (backend d window).scalar.negInf=
      (backend d window).scalar.negInf := rfl

theorem fixed_choices (d window : Nat) (p : Option ComputeDType) (dtype : FloatDType)
    (a a' b b' e e' : Expr) (ha : form a=form a') (hb : form b=form b') (he : form e=form e')
    (edge : Edge) (thread : Owner) :
    (plan d window).reverseMul p dtype a b thread=(plan d window).reverseMul p dtype a' b' thread ∧
    (plan d window).route e edge thread=(plan d window).route e' edge thread ∧ owner d e=owner d e' := by
  simp only [plan,owner,ha,hb,he,and_self]

theorem leaf_payload_ignored (dtype : FloatDType) (a b : Word) : form (.leaf dtype a)=form (.leaf dtype b) := rfl
theorem literal_payload_ignored (p : Option ComputeDType) (dtype : FloatDType) (a b : ℝ) :
    form (.literal p dtype a)=form (.literal p dtype b) ∧ live (.literal p dtype a)=live (.literal p dtype b) := ⟨rfl,rfl⟩
theorem live_product (p : Option ComputeDType) (dtype : FloatDType) (a b : Expr) :
    live (.product p dtype a b)=Bool.true := rfl

/-- A window=1 descriptor carries the exact emitted maximum tree, followed
by provenance for source lanes replaced by compiler constants. The tag and
omitted-lane annotation are syntax only. Every max node inside the emitted
code is executed, including max(S,S) reached through runtime masks. -/
def emittedMaximum {n : Nat} (maximum : Word → Word → Word)
    (literal : Word → Word) (inputs : Fin n → Word) : NativeGraph.Tree n → Word
  | .node (.padding _) (.node code _) => evalTree maximum literal inputs code
  | tree => evalTree maximum literal inputs tree

theorem emittedMaximum_code {n : Nat} (maximum : Word → Word → Word)
    (literal : Word → Word) (inputs : Fin n → Word) (tag : Word)
    (code omitted : NativeGraph.Tree n) :
    emittedMaximum maximum literal inputs (.node (.padding tag) (.node code omitted))=
      evalTree maximum literal inputs code := rfl

/-- Native word evaluation uses only local scalar ports and explicit fixed K8 instruction chains. -/
def evaluate (ops : ScalarOps) (width window : Nat) : Expr → Owner → Word
  | .leaf d w,_ => (primitives ops).leaf d w
  | .literal p d x,_ => (primitives ops).literal p d x
  | .fromNat p x,_ => (primitives ops).fromNat p x
  | .fromInt p x,_ => (primitives ops).fromInt p x
  | e@(.cast p source target x),owner =>
      (primitives ops).cast p source target (evaluate ops width window x ((plan width window).route e .unary owner))
  | e@(.unary p op x),owner => (primitives ops).unary p op (evaluate ops width window x ((plan width window).route e .unary owner))
  | e@(.binary p d op x y),owner => (primitives ops).binary p d op
      (evaluate ops width window x ((plan width window).route e .left owner)) (evaluate ops width window y ((plan width window).route e .right owner))
  | e@(.product p d x y),owner =>
      let a := evaluate ops width window x ((plan width window).route e .left owner)
      let b := evaluate ops width window y ((plan width window).route e .right owner)
      if (plan width window).reverseMul p d x y owner then (primitives ops).binary p d .mul b a else (primitives ops).binary p d .mul a b
  | e@(.fma p d x y acc),owner =>
      let a := evaluate ops width window x ((plan width window).route e .factorLeft owner)
      let b := evaluate ops width window y ((plan width window).route e .factorRight owner)
      let c := evaluate ops width window acc ((plan width window).route e .accumulator owner)
      if (plan width window).reverseFMA p d x y owner then (primitives ops).fma p d b a c else (primitives ops).fma p d a b c
  | e@(.guard p d value test replacement),owner =>
      let x := evaluate ops width window value ((plan width window).route e .guardValue owner)
      if (primitives ops).eq p d x (evaluate ops width window test ((plan width window).route e .guardTest owner)) then
        evaluate ops width window replacement ((plan width window).route e .guardReplacement owner) else x
  | e@(.reduction p d op n trees inputs),owner =>
      let values := fun i => evaluate ops width window (inputs i) ((plan width window).route e (.reductionInput i.val) owner)
      if window=1 ∧ op=.max then
        emittedMaximum ((primitives ops).binary p d op) ((primitives ops).leaf d) values (trees owner).tree
      else evalTree ((primitives ops).binary p d op) ((primitives ops).leaf d) values (trees owner).tree
  | e@(.pendingDot p l r out opts batch m k n a b i),owner =>
      dotAt ops batch
        (fun j => evaluate ops width window (a j) ((plan width window).route e (.dotLeft (TileShape.linearIndex _ j)) owner))
        (fun j => evaluate ops width window (b j) ((plan width window).route e (.dotRight (TileShape.linearIndex _ j)) owner))
        Bool.false ((primitives ops).literal p out 0) i
  | e@(.dotAcc p l r out opts batch m k n a b acc i),owner =>
      dotAt ops batch
        (fun j => evaluate ops width window (a j) ((plan width window).route e (.dotLeft (TileShape.linearIndex _ j)) owner))
        (fun j => evaluate ops width window (b j) ((plan width window).route e (.dotRight (TileShape.linearIndex _ j)) owner))
        (live acc) (evaluate ops width window acc ((plan width window).route e .accumulator owner)) i

#axiomsClean emittedMaximum_code
#axiomsClean initializer_fold
#axiomsClean fixed_choices
#axiomsClean literal_payload_ignored
#axiomsClean live_product
end
end VeriTile.Bench.Repairs.ContextAttnMistral.KernelNative
