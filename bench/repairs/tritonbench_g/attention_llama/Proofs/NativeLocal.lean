import bench.repairs.tritonbench_g.attention_llama.Proofs.NativeGraph

/-! Exact local numerical obligations at the operands reached by the recorded
native graph. A term records one primitive call, never a kernel result law. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.NativeLocal

inductive Term (α : Type) where
  | atom (value : α)
  | zero | one | negativeInfinity | log2e
  | add (x y : Term α) | sub (x y : Term α) | mul (x y : Term α) | div (x y : Term α)
  | maximum (x y : Term α)
  | ex2Approx (x : Term α) | half (x : Term α)
  | decode (byte : UInt8)
  | fma (x y z : Term α)
  | mma16 (x y : Fin 16 → Term α) (z : Term α)

def termOps (α : Type) : NativeGraph.Ops (Term α) where
  zero := .zero
  one := .one
  negativeInfinity := .negativeInfinity
  log2e := .log2e
  add := .add
  sub := .sub
  mul := .mul
  div := .div
  maximum := .maximum
  ex2Approx := .ex2Approx
  half := .half
  e5m2ToHalf := .decode
  fma := .fma
  mma16 := .mma16

def eval {α β : Type} (A : NativeGraph.Ops β) (leaf : α → β) : Term α → β
  | .atom x => leaf x
  | .zero => A.zero
  | .one => A.one
  | .negativeInfinity => A.negativeInfinity
  | .log2e => A.log2e
  | .add x y => A.add (eval A leaf x) (eval A leaf y)
  | .sub x y => A.sub (eval A leaf x) (eval A leaf y)
  | .mul x y => A.mul (eval A leaf x) (eval A leaf y)
  | .div x y => A.div (eval A leaf x) (eval A leaf y)
  | .maximum x y => A.maximum (eval A leaf x) (eval A leaf y)
  | .ex2Approx x => A.ex2Approx (eval A leaf x)
  | .half x => A.half (eval A leaf x)
  | .decode byte => A.e5m2ToHalf byte
  | .fma x y z => A.fma (eval A leaf x) (eval A leaf y) (eval A leaf z)
  | .mma16 x y z => A.mma16 (fun i => eval A leaf (x i)) (fun i => eval A leaf (y i)) (eval A leaf z)

/-- Each equation concerns exactly one primitive and its actual evaluated
operands. FMA and the fixed m16n8k16 ISA lane port have separate obligations. Half casts and E5M2 decode
remain explicit; no rounded mul-add equality or attention formula is assumed. -/
def Local {α β : Type} (A : NativeGraph.Ops α) (B : NativeGraph.Ops β) (f : α → β) : Term α → Prop
  | .atom _ => True
  | .zero => f A.zero = B.zero
  | .one => f A.one = B.one
  | .negativeInfinity => f A.negativeInfinity = B.negativeInfinity
  | .log2e => f A.log2e = B.log2e
  | .add x y => Local A B f x ∧ Local A B f y ∧
      f (A.add (eval A id x) (eval A id y)) = B.add (f (eval A id x)) (f (eval A id y))
  | .sub x y => Local A B f x ∧ Local A B f y ∧
      f (A.sub (eval A id x) (eval A id y)) = B.sub (f (eval A id x)) (f (eval A id y))
  | .mul x y => Local A B f x ∧ Local A B f y ∧
      f (A.mul (eval A id x) (eval A id y)) = B.mul (f (eval A id x)) (f (eval A id y))
  | .div x y => Local A B f x ∧ Local A B f y ∧
      f (A.div (eval A id x) (eval A id y)) = B.div (f (eval A id x)) (f (eval A id y))
  | .maximum x y => Local A B f x ∧ Local A B f y ∧
      f (A.maximum (eval A id x) (eval A id y)) = B.maximum (f (eval A id x)) (f (eval A id y))
  | .ex2Approx x => Local A B f x ∧ f (A.ex2Approx (eval A id x)) = B.ex2Approx (f (eval A id x))
  | .half x => Local A B f x ∧ f (A.half (eval A id x)) = B.half (f (eval A id x))
  | .decode byte => f (A.e5m2ToHalf byte) = B.e5m2ToHalf byte
  | .fma x y z => Local A B f x ∧ Local A B f y ∧ Local A B f z ∧
      f (A.fma (eval A id x) (eval A id y) (eval A id z)) =
        B.fma (f (eval A id x)) (f (eval A id y)) (f (eval A id z))
  | .mma16 x y z => (∀ i, Local A B f (x i)) ∧ (∀ i, Local A B f (y i)) ∧ Local A B f z ∧
      f (A.mma16 (fun i => eval A id (x i)) (fun i => eval A id (y i)) (eval A id z)) =
        B.mma16 (fun i => f (eval A id (x i))) (fun i => f (eval A id (y i))) (f (eval A id z))

theorem eval_transport {α β : Type} (A : NativeGraph.Ops α) (B : NativeGraph.Ops β)
    (f : α → β) (e : Term α) (h : Local A B f e) : f (eval A id e) = eval B f e := by
  induction e with
  | atom x => rfl
  | zero => exact h
  | one => exact h
  | negativeInfinity => exact h
  | log2e => exact h
  | decode byte => exact h
  | add x y ix iy | sub x y ix iy | mul x y ix iy | div x y ix iy | maximum x y ix iy =>
    simpa only [eval,ix h.1,iy h.2.1] using h.2.2
  | ex2Approx x ix | half x ix => simpa only [eval,ix h.1] using h.2
  | fma x y z ix iy iz =>
    simpa only [eval,ix h.1,iy h.2.1,iz h.2.2.1] using h.2.2.2
  | mma16 x y z ix iy iz =>
    simpa only [eval,funext (fun i => ix i (h.1 i)),funext (fun i => iy i (h.2.1 i)),iz h.2.2.1]
      using h.2.2.2

theorem evaluation {α β : Type} (A : NativeGraph.Ops β) (leaf : α → β) :
    NativeGraph.Agreement (termOps α) A (eval A leaf) := by
  constructor <;> intros <;> rfl

def outputTerm {α : Type} {D : Nat} (input : SourceGraph.Inputs α D) (scale : α)
    (fp8 causal : Bool) (N position pid lane : Nat) (d : Fin D) : Term α :=
  NativeGraph.returnedWord (termOps α) (input.map Term.atom) (Term.atom scale)
    fp8 causal N position pid lane d

theorem output_eval {α β : Type} {D : Nat} (A : NativeGraph.Ops β) (leaf : α → β)
    (input : SourceGraph.Inputs α D) (scale : α) (fp8 causal : Bool)
    (N position pid lane : Nat) (d : Fin D) :
    eval A leaf (outputTerm input scale fp8 causal N position pid lane d) =
      NativeGraph.returnedWord A (input.map leaf) (leaf scale) fp8 causal N position pid lane d := by
  exact NativeGraph.returnedWord_map (termOps α) A (eval A leaf) (evaluation A leaf)
    (input.map Term.atom) (Term.atom scale) fp8 causal N position pid lane d

theorem output_transport {α β : Type} {D : Nat} (A : NativeGraph.Ops α) (B : NativeGraph.Ops β)
    (f : α → β) (input : SourceGraph.Inputs α D) (scale : α) (fp8 causal : Bool)
    (N position pid lane : Nat) (d : Fin D)
    (h : Local A B f (outputTerm input scale fp8 causal N position pid lane d)) :
    f (NativeGraph.returnedWord A input scale fp8 causal N position pid lane d) =
      NativeGraph.returnedWord B (input.map f) (f scale) fp8 causal N position pid lane d := by
  have he := eval_transport A B f _ h
  rw [output_eval,output_eval] at he
  exact he

end VeriTile.Bench.Repairs.AttentionLlama.NativeLocal
