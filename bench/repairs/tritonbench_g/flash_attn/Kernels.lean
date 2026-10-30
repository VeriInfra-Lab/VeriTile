import bench.repairs.tritonbench_g.flash_attn.KernelOnly.SourceSyntax
import bench.repairs.tritonbench_g.flash_attn.Proofs.Reduction
namespace VeriTile.Bench.Repairs.FlashAttn
set_option maxHeartbeats 4000000
abbrev Word := BitVec 32
inductive Region where
  | q | k | v | o | l | other (id : Nat)
  deriving DecidableEq, Repr
abbrev Memory (α : Type) := Region → Nat → α
structure Shape where
  batches : Nat
  heads : Nat
  length : Nat
  dim : Nat
  causal : Bool
  deriving DecidableEq, Repr
structure Strides where
  batch : Nat
  head : Nat
  row : Nat
  dim : Nat
  deriving DecidableEq, Repr
structure Layout where
  shape : Shape
  q : Strides
  k : Strides
  v : Strides
  o : Strides
  /-- Finite structural serialization choice for redundant shared L stores. -/
  scratchOwner : Nat → Nat → Nat → Fin 4

def Shape.Valid (s : Shape) : Prop :=
  0<s.batches ∧ 0<s.heads ∧ 0<s.length ∧ s.dim∈[16,32,64,128]
def address (t : Strides) (b h n d : Nat) : Nat := b*t.batch+h*t.head+n*t.row+d*t.dim
def lAddress (s : Shape) (b h n : Nat) : Nat := (b*s.heads+h)*s.length+n
def ceil (n b : Nat) := (n+b-1)/b
def stop (s : Shape) (pid : Nat) : Nat := if s.causal then min ((pid+1)*128) s.length else s.length
def rounds (s : Shape) (pid : Nat) : Nat := ceil (stop s pid) 64
/-- Exactly the four bounded WGMMA component families seen in the eight pinned profiles. -/
inductive Width where
  | n16 | n32 | n64 | n128
  deriving DecidableEq, Repr

def width (d : Nat) : Width := if d=16 then .n16 else if d=32 then .n32 else if d=64 then .n64 else .n128
/-- Scalar leaves and one fixed K8 instruction component. No reduction, whole dot,
row value, source execution, address or final output is a numerical port. -/
structure Ops (α : Type) where
  zero : α
  negInf : α
  log2e : α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  max : α → α → α
  exp2 : α → α
  log2 : α → α
  div : α → α → α
  narrow : α → α
  widen : α → α
  mma8 : Width → (Fin 8 → α) → (Fin 8 → α) → α → α

def dot (p : Ops α) (w : Width) (k : Nat) (a b : Nat → α) (seed : α) : α :=
  (List.range (k/8)).foldl (fun acc chunk =>
    p.mma8 w (fun i => a (chunk*8+i.val)) (fun i => b (chunk*8+i.val)) acc) seed

def logical (m : Memory α) (r : Region) (t : Strides) (b h n d : Nat) : α :=
  m r (address t b h n d)
def query (x : Layout) (p : Ops α) (m : Memory α) (scale : α) (b h n e : Nat) : α :=
  p.widen (p.narrow (p.mul (p.mul scale p.log2e)
    (if n<x.shape.length then logical m .q x.q b h n e else p.zero)))
def key (x : Layout) (p : Ops α) (m : Memory α) (b h n e : Nat) : α :=
  if n<x.shape.length then logical m .k x.k b h n e else p.zero
def value (x : Layout) (p : Ops α) (m : Memory α) (b h n d : Nat) : α :=
  if n<x.shape.length then logical m .v x.v b h n d else p.zero

def visible (s : Shape) (row key : Nat) : Bool :=
  decide (key<s.length) && (!s.causal || decide (key≤row))
/-- The masked zero or negative-infinity tensor is the actual compiled dot accumulator seed.
The K8 chain retains the source q-scale-half-widen path and default TF32 lowering. -/
def score (x : Layout) (p : Ops α) (m : Memory α) (scale : α) (b h row keyRow : Nat) : α :=
  dot p .n64 x.shape.dim (query x p m scale b h row) (key x p m b h keyRow)
    (if visible x.shape row keyRow then p.zero else p.negInf)
structure Acc (α : Type) where
  maximum : Fin 4 → α
  denominator : Fin 4 → α
  output : α

def initial (p : Ops α) : Acc α := ⟨fun _=>p.negInf,fun _=>p.zero,p.zero⟩
/-- The four reduction replicas are explicit. Each probability uses the replica
that physically owns its key column, and each output seed uses its own column
replica. No equality of floating reduction orders is assumed. -/
def blockStep (x : Layout) (p : Ops α) (m : Memory α) (scale : α)
    (b h row col : Nat) (acc : Acc α) (block : Nat) : Acc α :=
  let scores := fun j => score x p m scale b h row (block*64+j)
  let mx := fun owner=>p.max (acc.maximum owner) ((Reduction.plan owner).eval p.max scores)
  let alpha := fun owner=>p.exp2 (p.sub (acc.maximum owner) (mx owner))
  let probability := fun j => p.exp2 (p.sub (scores j) (mx (Reduction.owner j)))
  let owner := Reduction.owner col
  let outScale := p.fma (acc.denominator owner) p.zero (alpha owner)
  let out := dot p (width x.shape.dim) 64 (fun j => p.widen (p.narrow (probability j)))
    (fun j => value x p m b h (block*64+j) col) (p.mul acc.output outScale)
  let den := fun owner=>p.fma (acc.denominator owner) (alpha owner)
    ((Reduction.plan owner).eval p.add probability)
  ⟨mx,den,out⟩

def accumulated (x : Layout) (p : Ops α) (m : Memory α) (scale : α)
    (b h row col : Nat) : Acc α :=
  (List.range (rounds x.shape (row/128))).foldl
    (blockStep x p m scale b h row col) (initial p)
def outputValue (x : Layout) (p : Ops α) (m : Memory α) (scale : α) (b h row col : Nat) : α :=
  let a := accumulated x p m scale b h row col
  p.widen (p.narrow (p.div a.output (a.denominator (Reduction.owner col))))
def logValue (x : Layout) (p : Ops α) (m : Memory α) (scale : α) (b h row : Nat) : α :=
  let a := accumulated x p m scale b h row 0
  p.add (a.maximum (x.scratchOwner b h row)) (p.log2 (a.denominator (x.scratchOwner b h row)))

inductive Work where
  | output (batch head row col : Nat)
  | logsum (batch head row : Nat)
  deriving DecidableEq, Repr

def Work.active (s : Shape) : Work → Prop
  | .output b h n d => b<s.batches ∧ h<s.heads ∧ n<s.length ∧ d<s.dim
  | .logsum b h n => b<s.batches ∧ h<s.heads ∧ n<s.length
instance (s : Shape) (w : Work) : Decidable (w.active s) := by cases w <;> unfold Work.active <;> infer_instance

def Work.target (x : Layout) : Work → Region × Nat
  | .output b h n d => (.o,address x.o b h n d)
  | .logsum b h n => (.l,lAddress x.shape b h n)
def Work.result (x : Layout) (p : Ops α) (m : Memory α) (scale : α) : Work → α
  | .output b h n d => outputValue x p m scale b h n d
  | .logsum b h n => logValue x p m scale b h n

def store (m : Memory α) (target : Region × Nat) (v : α) : Memory α :=
  fun r a => if (r,a)=target then v else m r a
def step (x : Layout) (p : Ops α) (scale : α) (w : Work) (m : Memory α) : Memory α :=
  store m (w.target x) (w.result x p m scale)
def execute (x : Layout) (p : Ops α) (scale : α) : List Work → Memory α → Memory α
  | [],m => m
  | w::ws,m => execute x p scale ws (step x p scale w m)
end VeriTile.Bench.Repairs.FlashAttn

namespace VeriTile.Bench.Repairs.FlashAttn
/-- Selected repaired Python body in the shared Triton source language. -/
abbrev actual_kernel := ActualSource.kernel
theorem shared_source (p : ActualSource.Parameters) : actual_kernel p=ActualSource.kernel p := rfl
#axiomsClean shared_source
end VeriTile.Bench.Repairs.FlashAttn
