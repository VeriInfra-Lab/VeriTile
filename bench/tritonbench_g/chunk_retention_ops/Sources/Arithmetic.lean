import bench.tritonbench_g.chunk_retention_ops.Sources.Geometry
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.Ring
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
open scoped BigOperators

/-- Local scalar primitives only: no dot, reduction, tensor or output evaluator. -/
structure Ops (α : Type) where
  zero : α
  integer : Int → α
  cast : α → α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  exp2 : α → α
  log2 : α → α

def chain (ops : Ops α) : Nat → (Nat → α) → (Nat → α) → α → α
  | 0,_,_,seed => seed
  | n+1,left,right,seed => ops.fma (left n) (right n) (chain ops n left right seed)

/-- Explicit outer tile loop and inner scalar FMA order. -/
def blockChain (ops : Ops α) (width : Nat) (left right : Nat → α) : Nat → α → α
  | 0,seed => seed
  | n+1,seed => chain ops width (fun i => left (n*width+i))
      (fun i => right (n*width+i)) (blockChain ops width left right n seed)

theorem chain_congr (ops : Ops α) (n : Nat) (l r l' r' : Nat → α) (seed : α)
    (hl : ∀ i, i<n → l i=l' i) (hr : ∀ i, i<n → r i=r' i) :
    chain ops n l r seed=chain ops n l' r' seed := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [chain,hl n (by omega),hr n (by omega)]
    rw [ih (fun i hi => hl i (by omega)) (fun i hi => hr i (by omega))]

inductive Expr where
  | word : BitVec 32 → Expr
  | zero : Expr
  | integer : Int → Expr
  | cast : Expr → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | fma : Expr → Expr → Expr → Expr
  | exp2 : Expr → Expr
  | log2 : Expr → Expr

def graphOps : Ops Expr := ⟨.zero,.integer,.cast,.add,.sub,.mul,.fma,.exp2,.log2⟩

/-- The transcendental functions name one fixed source/compiler profile.
In particular exp2 is not claimed to be correctly rounded and log2 includes
libdevice's internal FTZ instructions. -/
structure ISA where
  integer : Int → BitVec 32
  add : BitVec 32 → BitVec 32 → BitVec 32
  sub : BitVec 32 → BitVec 32 → BitVec 32
  mul : BitVec 32 → BitVec 32 → BitVec 32
  fma : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32
  exp2 : BitVec 32 → BitVec 32
  log2 : BitVec 32 → BitVec 32

def wordOps (isa : ISA) : Ops (BitVec 32) :=
  ⟨0,isa.integer,id,isa.add,isa.sub,isa.mul,isa.fma,isa.exp2,isa.log2⟩

def Expr.eval (isa : ISA) : Expr → BitVec 32
  | .word b => b
  | .zero => 0
  | .integer n => isa.integer n
  | .cast x => x.eval isa
  | .add x y => isa.add (x.eval isa) (y.eval isa)
  | .sub x y => isa.sub (x.eval isa) (y.eval isa)
  | .mul x y => isa.mul (x.eval isa) (y.eval isa)
  | .fma x y z => isa.fma (x.eval isa) (y.eval isa) (z.eval isa)
  | .exp2 x => isa.exp2 (x.eval isa)
  | .log2 x => isa.log2 (x.eval isa)

/-- Admitted operands may restrict NaN payloads without any FP=Real premise. -/
structure ScalarDomain where
  integer : Int → Prop
  unary : BitVec 32 → Prop
  binary : BitVec 32 → BitVec 32 → Prop
  ternary : BitVec 32 → BitVec 32 → BitVec 32 → Prop

def Expr.Admitted (isa : ISA) (d : ScalarDomain) : Expr → Prop
  | .word _ => True
  | .zero => True
  | .integer n => d.integer n
  | .cast x => x.Admitted isa d
  | .add x y | .sub x y | .mul x y =>
      x.Admitted isa d ∧ y.Admitted isa d ∧ d.binary (x.eval isa) (y.eval isa)
  | .fma x y z => x.Admitted isa d ∧ y.Admitted isa d ∧ z.Admitted isa d ∧
      d.ternary (x.eval isa) (y.eval isa) (z.eval isa)
  | .exp2 x | .log2 x => x.Admitted isa d ∧ d.unary (x.eval isa)

structure LocalConformance (native reference : ISA) (d : ScalarDomain) : Prop where
  integer : ∀ n, d.integer n → native.integer n=reference.integer n
  add : ∀ a b, d.binary a b → native.add a b=reference.add a b
  sub : ∀ a b, d.binary a b → native.sub a b=reference.sub a b
  mul : ∀ a b, d.binary a b → native.mul a b=reference.mul a b
  fma : ∀ a b c, d.ternary a b c → native.fma a b c=reference.fma a b c
  exp2 : ∀ a, d.unary a → native.exp2 a=reference.exp2 a
  log2 : ∀ a, d.unary a → native.log2 a=reference.log2 a

theorem Expr.transport (native reference : ISA) (d : ScalarDomain)
    (laws : LocalConformance native reference d) (x : Expr) (hx : x.Admitted native d) :
    x.eval native=x.eval reference := by
  induction x with
  | word b => rfl
  | zero => rfl
  | integer n => exact laws.integer n hx
  | cast x ih => exact ih hx
  | add x y ix iy => rw [eval,laws.add _ _ hx.2.2,ix hx.1,iy hx.2.1]; rfl
  | sub x y ix iy => rw [eval,laws.sub _ _ hx.2.2,ix hx.1,iy hx.2.1]; rfl
  | mul x y ix iy => rw [eval,laws.mul _ _ hx.2.2,ix hx.1,iy hx.2.1]; rfl
  | fma x y z ix iy iz =>
    rw [eval,laws.fma _ _ _ hx.2.2.2,ix hx.1,iy hx.2.1,iz hx.2.2.1]; rfl
  | exp2 x ih => rw [eval,laws.exp2 _ hx.2,ih hx.1]; rfl
  | log2 x ih => rw [eval,laws.log2 _ hx.2,ih hx.1]; rfl

def allOperands : ScalarDomain := ⟨fun _ => True,fun _ => True,fun _ _ => True,fun _ _ _ => True⟩
theorem all_admitted (isa : ISA) (x : Expr) : x.Admitted isa allOperands := by
  induction x <;> simp_all [Expr.Admitted,allOperands]
theorem reference_conforms (isa : ISA) (d : ScalarDomain) : LocalConformance isa isa d := by
  constructor <;> intros <;> rfl

#axiomsClean Expr.transport
#specNonCircular LocalConformance avoiding [chain,blockChain,Expr.transport]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

