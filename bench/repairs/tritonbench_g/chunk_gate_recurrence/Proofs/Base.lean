import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealMath
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ReductionPlan
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
open scoped BigOperators
noncomputable section

structure Shape where
  B : Nat
  H : Nat
  N : Nat
  KT : Nat
  VT : Nat
  bpos : 0 < B
  hpos : 0 < H
  kpos : 0 < KT
  vpos : 0 < VT

def Shape.heads (c : Shape) := c.B*c.H
def Shape.tiles (c : Shape) := c.KT*c.VT
def Shape.programs (c : Shape) := c.heads*c.tiles
def Shape.K (c : Shape) := c.KT*64
def Shape.V (c : Shape) := c.VT*16

theorem Shape.tiles_pos (c : Shape) : 0<c.tiles := Nat.mul_pos c.kpos c.vpos

/-- A normalized cell denotes one complete FP32 word in an ordinary allocation.
Physical addresses, actual seed strides and input aliasing are bound separately.
The fresh output/scratch tags are disjoint; caller-live cells include all gaps. -/
inductive Region where
  | x | gate | seed | incoming | output | dx | tileGate | dseed | dgate | caller
  deriving DecidableEq, Repr
abbrev Memory (α : Type) := Region → Nat → Nat → Nat → α
abbrev Tile (α : Type) := Nat → α

structure Ops (α : Type) where
  zero : α
  add : α → α → α
  mul : α → α → α
  fma : α → α → α → α

def realOps : Ops ℝ := ⟨0,(·+·),(·*·),fun a b c => a*b+c⟩
namespace Reduction
export VeriTile.Bench.Repairs.LayerNormLiger.ReductionPlan (Tree Plan)
def eval (ops : Ops α) (p : Plan n) (v : Fin n → α) := p.tree.eval ops.zero ops.add v

theorem real (p : Plan n) (v : Fin n → ℝ) : eval realOps p v=∑ i, v i := by
  change p.tree.eval 0 (·+·) v = _
  rw [VeriTile.Bench.Repairs.LayerNormLiger.ReductionPlan.Plan.real_sum]
  rw [← List.ofFn_eq_map, List.sum_ofFn]
/-- Complete product-reduction lowering. The fused node records the actual
compiler contraction of one product leaf into its scalar accumulator. -/
inductive ProductTree (n : Nat) where
  | zero
  | mul (index : Fin n)
  | add (left right : ProductTree n)
  | fma (index : Fin n) (rest : ProductTree n)

def ProductTree.leaves : ProductTree n → List (Fin n)
  | .zero => []
  | .mul i => [i]
  | .add l r => l.leaves ++ r.leaves
  | .fma i r => i :: r.leaves

def ProductTree.eval (ops : Ops α) (v : Fin n → α × α) : ProductTree n → α
  | .zero => ops.zero
  | .mul i => ops.mul (v i).1 (v i).2
  | .add l r => ops.add (l.eval ops v) (r.eval ops v)
  | .fma i r => ops.fma (v i).1 (v i).2 (r.eval ops v)

structure ProductPlan (n : Nat) where
  tree : ProductTree n
  coverage : tree.leaves.Perm (List.finRange n)

def product (ops : Ops α) (p : ProductPlan n) (v : Fin n → α × α) := p.tree.eval ops v

theorem ProductTree.real (p : ProductTree n) (v : Fin n → ℝ × ℝ) :
    p.eval realOps v=(p.leaves.map (fun i => (v i).1*(v i).2)).sum := by
  induction p with
  | zero => rfl
  | mul i => simp [eval,leaves,realOps]
  | add l r hl hr =>
    simp only [eval,hl,hr,leaves,List.map_append,List.sum_append]
    rfl
  | fma i r hr =>
    simp only [eval,hr,leaves,List.map_cons,List.sum_cons]
    rfl

theorem product_real (p : ProductPlan n) (v : Fin n → ℝ × ℝ) :
    product realOps p v=∑ i, (v i).1*(v i).2 := by
  rw [product,ProductTree.real,(p.coverage.map (fun i => (v i).1*(v i).2)).sum_eq,
    ← List.ofFn_eq_map,List.sum_ofFn]
end Reduction

structure Trees (c : Shape) where
  tile : Nat → Nat → Reduction.ProductPlan 1024
  host : Nat → Nat → Reduction.Plan c.tiles

namespace Memory
/-- One SIMD tile store; source lanes are 64 by 16, with no inactive lane. -/
def tile (m : Memory α) (region : Region) (program time : Nat) (v : Tile α) : Memory α :=
  fun r p t j => if r=region ∧ p=program ∧ t=time ∧ j<1024 then v j else m r p t j

def scalar (m : Memory α) (region : Region) (program time : Nat) (v : α) : Memory α :=
  fun r p t j => if r=region ∧ p=program ∧ t=time ∧ j=0 then v else m r p t j

@[simp] theorem tile_at (m : Memory α) (r : Region) (p t j : Nat) (v : Tile α) (hj : j<1024) :
    tile m r p t v r p t j=v j := by simp [tile,hj]
@[simp] theorem tile_region (m : Memory α) (r q : Region) (p t a b j : Nat) (v : Tile α) (h : q≠r) :
    tile m r p t v q a b j=m q a b j := by simp [tile,h]
@[simp] theorem scalar_at (m : Memory α) (r : Region) (p t : Nat) (v : α) :
    scalar m r p t v r p t 0=v := by simp [scalar]
@[simp] theorem scalar_region (m : Memory α) (r q : Region) (p t a b j : Nat) (v : α) (h : q≠r) :
    scalar m r p t v q a b j=m q a b j := by simp [scalar,h]

def map (f : α → β) (m : Memory α) : Memory β := fun r p t j => f (m r p t j)
end Memory

structure Inputs (α : Type) where
  x : Nat → Nat → Nat → α
  gate : Nat → Nat → α
  seed : Nat → Nat → α
  incoming : Nat → Nat → Nat → α

def inputsOf (m : Memory α) : Inputs α :=
  ⟨m .x,fun h t => m .gate h t 0,fun p j => m .seed p 0 j,m .incoming⟩

namespace Reference
/-- Input-only source graph, separate from source-memory execution and Real formulas. -/
def forward (ops : Ops α) (c : Shape) (a : Inputs α) (hasSeed : Bool) (p : Nat) : Nat → Tile α
  | 0 => fun j => if hasSeed then a.seed p j else ops.zero
  | t+1 => fun j => ops.fma (forward ops c a hasSeed p t j) (a.gate (p/c.tiles) t) (a.x p t j)

def reverse (ops : Ops α) (c : Shape) (a : Inputs α) (p : Nat) : Nat → Tile α
  | 0 => fun _ => ops.zero
  | n+1 => fun j => ops.fma (reverse ops c a p n j)
      (a.gate (p/c.tiles) (c.N-1-n)) (a.incoming p (c.N-1-n) j)

def dx (ops : Ops α) (c : Shape) (a : Inputs α) (p t j : Nat) :=
  if t+1<c.N then reverse ops c a p (c.N-1-t) j else ops.zero

def tileGate (ops : Ops α) (c : Shape) (trees : Trees c) (a : Inputs α)
    (hasSeed : Bool) (p t : Nat) :=
  if t+1<c.N then Reduction.product ops (trees.tile p t)
    (fun j => (reverse ops c a p (c.N-1-t) j,forward ops c a hasSeed p t j)) else ops.zero

def dgate (ops : Ops α) (c : Shape) (trees : Trees c) (a : Inputs α)
    (hasSeed : Bool) (h t : Nat) :=
  Reduction.eval ops (trees.host h t)
    (fun q => tileGate ops c trees a hasSeed (h*c.tiles+q) t)
end Reference

#specNonCircular Reference.forward avoiding [Memory.tile]
#specNonCircular Reference.reverse avoiding [Memory.tile]
#axiomsClean Reduction.real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
