import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Geometry
namespace VeriTile.Bench.Repairs.BlockSparseAttn
set_option maxHeartbeats 4000000
open scoped BigOperators
abbrev Memory (α : Type) := FlashAttn.Memory α
abbrev Word := FlashAttn.Word

/-- A finite scalar expression tree, not a numerical operation port. -/
inductive Tree (n : Nat) where
  | leaf (i : Fin n)
  | merge (left right : Tree n)
def Tree.leaves : Tree n → List (Fin n)
  | .leaf i=>[i]
  | .merge l r=>l.leaves++r.leaves
def Tree.eval (op : α → α → α) (f : Fin n → α) : Tree n → α
  | .leaf i=>f i
  | .merge l r=>op (l.eval op f) (r.eval op f)

/-- Source-specific compiler layout data. Each physical reduction replica
contains every key lane once. This contains no numeric functions or results. -/
structure Reduction (n : Nat) where
  replicas : Nat
  tree : Fin replicas → Tree n
  keyOwner : Fin n → Fin replicas
  outputOwner : Nat → Fin replicas
  coverage : ∀ owner,(tree owner).leaves.Perm (List.finRange n)

def owner (column : Nat) : Fin 4 := ⟨column%8/2,by have:=Nat.mod_lt column (by decide : 0<8);omega⟩

/-- Finite instruction families; each component consumes exactly eight pairs
and its live accumulator. No full dot or whole reduction is a numerical leaf. -/
inductive Instruction where
  | mma16n8k8 | wgmma64n16k8 | wgmma64n32k8 | wgmma64n64k8
  | wgmma64n128k8 | wgmma64n256k8
  deriving DecidableEq
structure Ops (α : Type) where
  zero : α
  negInf : α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  max : α → α → α
  exp : α → α
  div : α → α → α
  mma8 : Instruction → (Fin 8 → α) → (Fin 8 → α) → α → α

structure Layout where
  shape : Shape
  q : Strides
  k : Strides
  v : Strides
  csr : CSR
  evenM : Bool
  evenN : Bool
  reduction : Reduction shape.blockN
  scoreInstruction : Instruction
  valueInstruction : Instruction

def dot (p : Ops α) (instruction : Instruction) (n : Nat)
    (a b : Nat → α) (seed : α) : α :=
  (List.range (n/8)).foldl (fun acc chunk=>p.mma8 instruction
    (fun i=>a (chunk*8+i.val)) (fun i=>b (chunk*8+i.val)) acc) seed

def qLoad (x : Layout) (p : Ops α) (m : Memory α) (b h row col : Nat) : α :=
  if x.evenM || decide (row<x.shape.length) then m .q (address x.q b h row col) else p.zero
def kLoad (x : Layout) (p : Ops α) (m : Memory α) (b h : Nat) (row : Int) (col : Nat) : α :=
  if x.evenN || decide (row<(x.shape.length:Int)) then
    m .k (signedAddress x.k b (kvHead x.shape h) row col).toNat else p.zero
def vLoad (x : Layout) (p : Ops α) (m : Memory α) (b h : Nat) (row : Int) (col : Nat) : α :=
  if x.evenN || decide (row<(x.shape.length:Int)) then
    m .v (signedAddress x.v b (kvHead x.shape h) row col).toNat else p.zero

def scoreDot (x : Layout) (p : Ops α) (m : Memory α) (b h row : Nat) (key : Int) : α :=
  (List.range x.shape.blocks).foldl (fun acc block=>
    dot p x.scoreInstruction x.shape.blockD
      (fun d=>qLoad x p m b h row (block*x.shape.blockD+d))
      (fun d=>kLoad x p m b h key (block*x.shape.blockD+d)) acc) p.zero

/-- PTX fuses scaled score plus causal zero/negative-infinity mask. -/
def score (x : Layout) (p : Ops α) (m : Memory α) (scale : α)
    (b h row visit : Nat) (lane : Fin x.shape.blockN) : α :=
  let key := x.csr.key x.shape h row visit lane.val
  p.fma scale (scoreDot x p m b h row key) (if key≤(row:Int) then p.zero else p.negInf)

structure Acc (α : Type) (replicas : Nat) where
  maximum : Fin replicas → α
  denominator : Fin replicas → α
  output : α
def initial (p : Ops α) (replicas : Nat) : Acc α replicas := ⟨fun _=>p.negInf,fun _=>p.zero,p.zero⟩

/-- Retained source normalization, with every physical row replica explicit.
The cast to Q's FP32 storage type is the identity on selected raw words. -/
def normalizedStep (plan : Reduction n) (instruction : Instruction) (p : Ops α)
    (scores : Fin n → α) (values : Nat → α) (col : Nat)
    (acc : Acc α plan.replicas) : Acc α plan.replicas :=
  let blockMax := fun o=>(plan.tree o).eval p.max scores
  let probability := fun j=>p.exp (p.sub (scores j) (blockMax (plan.keyOwner j)))
  let blockSum := fun o=>(plan.tree o).eval p.add probability
  let nextMax := fun o=>p.max (acc.maximum o) (blockMax o)
  let alpha := fun o=>p.exp (p.sub (acc.maximum o) (nextMax o))
  let beta := fun o=>p.exp (p.sub (blockMax o) (nextMax o))
  let nextDen := fun o=>p.fma (acc.denominator o) (alpha o) (p.mul (blockSum o) (beta o))
  let scaledProb := fun j=>p.mul (probability j) (p.div (beta (plan.keyOwner j)) (nextDen (plan.keyOwner j)))
  let o := plan.outputOwner col
  let scaleAcc := p.mul (alpha o) (p.div (acc.denominator o) (nextDen o))
  let out := dot p instruction n
    (fun j=>if hj : j<n then scaledProb ⟨j,hj⟩ else p.zero) values
    (p.mul acc.output scaleAcc)
  ⟨nextMax,nextDen,out⟩

def blockStep (x : Layout) (p : Ops α) (m : Memory α) (scale : α)
    (b h row col : Nat) (acc : Acc α x.reduction.replicas) (visit : Nat) : Acc α x.reduction.replicas :=
  normalizedStep x.reduction x.valueInstruction p (score x p m scale b h row visit)
    (fun j=>vLoad x p m b h (x.csr.key x.shape h row visit j) col) col acc

def rowValue (x : Layout) (p : Ops α) (m : Memory α) (scale : α) (w : Work) : α :=
  ((List.range (x.csr.count x.shape w.head w.row)).foldl
    (blockStep x p m scale w.batch w.head w.row w.col) (initial p x.reduction.replicas)).output

def step (x : Layout) (p : Ops α) (scale : α) (w : Work) (m : Memory α) : Memory α :=
  FlashAttn.store m (.o,w.target x.shape) (rowValue x p m scale w)
def execute (x : Layout) (p : Ops α) (scale : α) : List Work → Memory α → Memory α
  | [],m=>m
  | w::ws,m=>execute x p scale ws (step x p scale w m)

end VeriTile.Bench.Repairs.BlockSparseAttn
