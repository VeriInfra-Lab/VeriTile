import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Geometry
import Mathlib.Tactic.Ring
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section
open scoped BigOperators

/-- One scalar operation at a time. No dot or aggregate result port. -/
structure Ops (α : Type) where
  zero : α
  cast : α → α
  mul : α → α → α
  fma : α → α → α → α

def chain (ops : Ops α) : Nat → (Nat → α) → (Nat → α) → α → α
  | 0,_,_,seed => seed
  | n+1,left,right,seed => ops.fma (left n) (right n) (chain ops n left right seed)

def realOps : Ops ℝ := ⟨0,id,(·*·),fun a b c => a*b+c⟩

theorem chain_real (n : Nat) (left right : Nat → ℝ) (seed : ℝ) :
    chain realOps n left right seed=seed+∑ i ∈ Finset.range n, left i*right i := by
  induction n with
  | zero => simp [chain]
  | succ n ih =>
    change left n*right n+chain realOps n left right seed=_
    rw [ih,Finset.sum_range_succ]
    ring

theorem chain_congr (ops : Ops α) (n : Nat) (left right left' right' : Nat → α)
    (seed : α) (hl : ∀ i, i<n → left i=left' i) (hr : ∀ i, i<n → right i=right' i) :
    chain ops n left right seed=chain ops n left' right' seed := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [chain,hl n (by omega),hr n (by omega)]
    rw [ih (fun i hi => hl i (by omega)) (fun i hi => hr i (by omega))]

/-- FP32 storage casts are retained as nodes; their native word action is
identity because both source and destination are already FP32. -/
inductive Expr where
  | word : BitVec 32 → Expr
  | zero : Expr
  | cast32 : Expr → Expr
  | mulRN : Expr → Expr → Expr
  | fmaRN : Expr → Expr → Expr → Expr

def graphOps : Ops Expr := ⟨.zero,.cast32,.mulRN,.fmaRN⟩

structure ISA where
  mulRN : BitVec 32 → BitVec 32 → BitVec 32
  fmaRN : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32

def Expr.eval (isa : ISA) : Expr → BitVec 32
  | .word b => b
  | .zero => 0
  | .cast32 x => x.eval isa
  | .mulRN x y => isa.mulRN (x.eval isa) (y.eval isa)
  | .fmaRN x y z => isa.fmaRN (x.eval isa) (y.eval isa) (z.eval isa)

def Expr.real (decode : BitVec 32 → ℝ) : Expr → ℝ
  | .word b => decode b
  | .zero => 0
  | .cast32 x => x.real decode
  | .mulRN x y => x.real decode*y.real decode
  | .fmaRN x y z => x.real decode*y.real decode+z.real decode

structure NumericDomain where
  mul : BitVec 32 → BitVec 32 → Prop
  fma : BitVec 32 → BitVec 32 → BitVec 32 → Prop

/-- Three local families. The wrapper's one scaleWord equation is separate.
Domains restrict obligations to admitted scalar operands, not final results. -/
structure LocalLaws (isa : ISA) (decode : BitVec 32 → ℝ) (dom : NumericDomain) : Prop where
  zeroWord : decode 0=0
  mulRN : ∀ a b, dom.mul a b → decode (isa.mulRN a b)=decode a*decode b
  fmaRN : ∀ a b c, dom.fma a b c → decode (isa.fmaRN a b c)=decode a*decode b+decode c

def Expr.Admitted (isa : ISA) (dom : NumericDomain) : Expr → Prop
  | .word _ => True
  | .zero => True
  | .cast32 x => x.Admitted isa dom
  | .mulRN x y => x.Admitted isa dom ∧ y.Admitted isa dom ∧ dom.mul (x.eval isa) (y.eval isa)
  | .fmaRN x y z => x.Admitted isa dom ∧ y.Admitted isa dom ∧ z.Admitted isa dom ∧
      dom.fma (x.eval isa) (y.eval isa) (z.eval isa)

theorem Expr.transport (isa : ISA) (decode : BitVec 32 → ℝ) (dom : NumericDomain)
    (laws : LocalLaws isa decode dom) (x : Expr) (hx : x.Admitted isa dom) :
    decode (x.eval isa)=x.real decode := by
  induction x with
  | word b => rfl
  | zero => exact laws.zeroWord
  | cast32 x ih => exact ih hx
  | mulRN x y ihx ihy =>
    rw [Expr.eval,laws.mulRN _ _ hx.2.2,ihx hx.1,ihy hx.2.1]
    rfl
  | fmaRN x y z ihx ihy ihz =>
    rw [Expr.eval,laws.fmaRN _ _ _ hx.2.2.2,ihx hx.1,ihy hx.2.1,ihz hx.2.2.1]
    rfl

theorem chain_graph_real (decode : BitVec 32 → ℝ) (n : Nat)
    (left right : Nat → Expr) (seed : Expr) :
    (chain graphOps n left right seed).real decode=
      chain realOps n (fun i => (left i).real decode) (fun i => (right i).real decode) (seed.real decode) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    exact congrArg (fun t => (left n).real decode*(right n).real decode+t) ih

#axiomsClean chain_real
#axiomsClean Expr.transport
#specNonCircular LocalLaws avoiding [chain,Expr.transport]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
