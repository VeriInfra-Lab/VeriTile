import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Reduction
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Library

namespace VeriTile.Bench.Repairs.CrossEntropy2.Floating
open Source Reduction
noncomputable section
abbrev Word := BitVec 32

/-- The arithmetic graph extends the reused source-algebra interface with
the candidate's actual separate host multiplication. No device multiply is
substituted for that Torch operation. -/
inductive Expr where
  | raw (bits : Word)
  | literal (value : ℝ)
  | natural (value : Nat)
  | negativeInfinity
  | barrier (value : Expr)
  | binary (operation : Binary32) (left right : Expr)
  | unary (operation : Unary32) (value : Expr)
  | sum (width : Nat) (values : Fin width → Expr)
  | maximum (width : Nat) (values : Fin width → Expr)
  | negation (value : Expr)
  | hostAdd (left right : Expr)
  | hostMultiply (left right : Expr)
  | hostSquare (value : Expr)
  | hostScale (scale : ℝ) (value : Expr)
  | hostSum (width : Nat) (values : Fin width → Expr)
  | hostLSE (width : Nat) (values : Fin width → Expr)
  | rankSum (width : Nat) (values : Fin width → Expr)

instance : Inhabited Expr := ⟨.raw 0⟩

def algebra : FP32 Expr where
  literal := .literal
  fromNat := .natural
  negInf := .negativeInfinity
  load := .barrier
  add := .binary .add
  sub := .binary .sub
  mul := .binary .mul
  div := .binary .div
  exp := .unary .exp
  log := .unary .log
  sum := fun {width} values => .sum width values
  maximum := fun {width} values => .maximum width values

def hostArithmetic : Host.Arithmetic Expr where
  zero := .literal 0
  add := fun a b => .hostAdd (.barrier a) (.barrier b)
  multiply := fun a b => .hostMultiply (.barrier a) (.barrier b)
  square := fun x => .hostSquare (.barrier x)
  scale := fun scale x => .hostScale scale (.barrier x)
  sum := fun xs => .hostSum xs.length (fun i => .barrier (xs.get i))
  logsumexp := fun xs => .hostLSE xs.length (fun i => .barrier (xs.get i))
  rankSum := fun xs => .rankSum xs.length (fun i => .barrier (xs.get i))

/-- All deferred laws are individual scalar ports. There is no sum, max,
logsumexp, output, execution or successful-launch numerical oracle. -/
@[ext] structure Ports where
  literal : ℝ → Word
  natural : Nat → Word
  negativeInfinity : Word
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  div : Word → Word → Word
  neg : Word → Word
  fma : Word → Word → Word → Word
  exp : Word → Word
  log : Word → Word
  max : Word → Word → Word
  isInf : Word → Bool
  hostAdd : Word → Word → Word
  hostMul : Word → Word → Word
  hostSub : Word → Word → Word
  hostExp : Word → Word
  hostLog : Word → Word
  hostMax : Word → Word → Word
  hostIsInf : Word → Bool

/-- Exact native reduction trees are profile data, with a checked complete
leaf permutation. Device/thread shape may choose different trees, without
changing the independent mathematical target. -/
structure Profile where
  sumTree : ∀ width, 0<width → Plan width
  maxTree : ∀ width, 0<width → Plan width
  hostSumTree : ∀ width, 0<width → Plan width
  hostMaxTree : ∀ width, 0<width → Plan width
  /-- logsumexp's exponential temporary is a distinct reduction call. -/
  hostLSESumTree : ∀ width, 0<width → Plan width := hostSumTree

def reduce (choice : ∀ width,0<width → Plan width) (op : Word → Word → Word)
    (empty : Word) (xs : List Word) : Word :=
  if positive : 0<xs.length then (choice xs.length positive).tree.run op empty (fun i => xs[i]?.getD empty) else empty

structure Evaluated where
  value : Word
  product : Option (Word × Word) := none

def add (n : Ports) (a b : Evaluated) : Evaluated :=
  match b.product with
  | some (x,y) => ⟨n.fma x y a.value,none⟩
  | none => match a.product with
    | some (x,y) => ⟨n.fma x y b.value,none⟩
    | none => ⟨n.add a.value b.value,none⟩

def sub (n : Ports) (a b : Evaluated) : Evaluated :=
  match b.product with
  | some (x,y) => ⟨n.fma (n.neg x) y a.value,none⟩
  | none => ⟨n.sub a.value b.value,none⟩

/-- Torch logsumexp's ordinary stable decomposition. Infinite maxima are
replaced before shifting; NaN propagation is retained by scalar ports. -/
def hostLSE (n : Ports) (profile : Profile) (xs : List Word) : Word :=
  let max := reduce profile.hostMaxTree n.hostMax n.negativeInfinity xs
  let shift := if n.hostIsInf max then n.literal 0 else max
  let exponentials := xs.map (fun x => n.hostExp (n.hostSub x shift))
  n.hostAdd (n.hostLog (reduce profile.hostLSESumTree n.hostAdd (n.literal 0) exponentials)) shift

def evaluate (n : Ports) (profile : Profile) : Expr → Evaluated
  | .raw x => ⟨x,none⟩
  | .literal x => ⟨n.literal x,none⟩
  | .natural x => ⟨n.natural x,none⟩
  | .negativeInfinity => ⟨n.negativeInfinity,none⟩
  | .barrier x => ⟨(evaluate n profile x).value,none⟩
  | .binary op a b =>
    let left := evaluate n profile a
    let right := evaluate n profile b
    match op with
    | .add => add n left right
    | .sub => sub n left right
    | .mul => ⟨n.mul left.value right.value,some (left.value,right.value)⟩
    | .div => ⟨n.div left.value right.value,none⟩
  | .unary op x =>
    match op with
    | .exp => ⟨n.exp (evaluate n profile x).value,none⟩
    | .log => ⟨n.log (evaluate n profile x).value,none⟩
  | .sum _ xs => ⟨reduce profile.sumTree n.add (n.literal 0) (List.ofFn (fun i => (evaluate n profile (xs i)).value)),none⟩
  | .maximum _ xs => ⟨reduce profile.maxTree n.max n.negativeInfinity (List.ofFn (fun i => (evaluate n profile (xs i)).value)),none⟩
  | .negation x => ⟨n.neg (evaluate n profile x).value,none⟩
  | .hostAdd x y => ⟨n.hostAdd (evaluate n profile x).value (evaluate n profile y).value,none⟩
  | .hostMultiply x y => ⟨n.hostMul (evaluate n profile x).value (evaluate n profile y).value,none⟩
  | .hostSquare x => let a := (evaluate n profile x).value; ⟨n.hostMul a a,none⟩
  | .hostScale scale x => ⟨n.hostMul (n.literal scale) (evaluate n profile x).value,none⟩
  | .hostSum _ xs | .rankSum _ xs =>
    ⟨reduce profile.hostSumTree n.hostAdd (n.literal 0) (List.ofFn (fun i => (evaluate n profile (xs i)).value)),none⟩
  | .hostLSE _ xs => ⟨hostLSE n profile (List.ofFn (fun i => (evaluate n profile (xs i)).value)),none⟩

def word (n : Ports) (profile : Profile) (expression : Expr) : Word :=
  (evaluate n profile expression).value

/-- Each equality is one scalar instruction at the selected FP32/default-FMA
profile. Full binary trees and host LSE are derived in `evaluate`. -/
structure Obligations (native reference : Ports) : Prop where
  literal : ∀ x, native.literal x=reference.literal x
  natural : ∀ x, native.natural x=reference.natural x
  negativeInfinity : native.negativeInfinity=reference.negativeInfinity
  add : ∀ x y, native.add x y=reference.add x y
  sub : ∀ x y, native.sub x y=reference.sub x y
  mul : ∀ x y, native.mul x y=reference.mul x y
  div : ∀ x y, native.div x y=reference.div x y
  neg : ∀ x, native.neg x=reference.neg x
  fma : ∀ x y z, native.fma x y z=reference.fma x y z
  exp : ∀ x, native.exp x=reference.exp x
  log : ∀ x, native.log x=reference.log x
  max : ∀ x y, native.max x y=reference.max x y
  isInf : ∀ x, native.isInf x=reference.isInf x
  hostAdd : ∀ x y, native.hostAdd x y=reference.hostAdd x y
  hostMul : ∀ x y, native.hostMul x y=reference.hostMul x y
  hostSub : ∀ x y, native.hostSub x y=reference.hostSub x y
  hostExp : ∀ x, native.hostExp x=reference.hostExp x
  hostLog : ∀ x, native.hostLog x=reference.hostLog x
  hostMax : ∀ x y, native.hostMax x y=reference.hostMax x y
  hostIsInf : ∀ x, native.hostIsInf x=reference.hostIsInf x

theorem ports_equal (native reference : Ports) (h : Obligations native reference) : native=reference := by
  apply Ports.ext
  all_goals first | exact h.negativeInfinity | exact funext h.literal | exact funext h.natural |
    exact funext (fun x => funext (h.add x)) | exact funext (fun x => funext (h.sub x)) |
    exact funext (fun x => funext (h.mul x)) | exact funext (fun x => funext (h.div x)) |
    exact funext h.neg | exact funext (fun x => funext (fun y => funext (h.fma x y))) |
    exact funext h.exp | exact funext h.log | exact funext (fun x => funext (h.max x)) |
    exact funext h.isInf | exact funext (fun x => funext (h.hostAdd x)) |
    exact funext (fun x => funext (h.hostMul x)) | exact funext (fun x => funext (h.hostSub x)) |
    exact funext h.hostExp | exact funext h.hostLog | exact funext (fun x => funext (h.hostMax x)) | exact funext h.hostIsInf

theorem complete_expression (native reference : Ports) (h : Obligations native reference)
    (profile : Profile) (expression : Expr) :
    evaluate native profile expression=evaluate reference profile expression := by rw [ports_equal native reference h]

#axiomsClean complete_expression
#specNonCircular Obligations avoiding [Source.forward,Source.backward,Source.values,Source.lossZ,Source.backwardValues]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Floating
