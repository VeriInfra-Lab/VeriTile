import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.Memory
import bench.repairs.tritonbench_g.triton_attention.Whole001.Proofs.Reduction

namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
open Memory
set_option maxHeartbeats 4000000

inductive Width where | n16 | n32 | n64 | n128 deriving DecidableEq

def width (d : Nat) : Width :=
  if d = 16 then .n16 else if d = 32 then .n32 else if d = 64 then .n64 else .n128

/-- Fixed scalar opcodes and one output component of a fixed K8 WGMMA.
The Bool records the PTX scale-D/seed-enable operand. There is no variable-
length dot, reduction, row, stage, memory or final-output numerical port. -/
structure Ops (α : Type) where
  zero : α
  one : α
  negInf : α
  log2e : α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  neg : α → α
  max : α → α → α
  exp2 : α → α
  div : α → α → α
  narrow : α → α
  widen : α → α
  mma8 : Width → (Fin 8 → α) → (Fin 8 → α) → Bool → α → α

/-- Redundant row stores are structural compiler choices, retained separately
for L and M. Preprocess Delta similarly records its selected reduction replica. -/
structure Owners where
  l : Nat → Nat → Fin 4
  m : Nat → Nat → Fin 4
  delta : Nat → Nat → Nat

def exp (p : Ops α) (x : α) : α := p.exp2 (p.mul x p.log2e)
def half (p : Ops α) (x : α) : α := p.widen (p.narrow x)

def dot (p : Ops α) (w : Width) (k : Nat) (a b : Nat → α) (live : Bool) (seed : α) : α :=
  (List.range (k / 8)).foldl (fun acc chunk =>
    p.mma8 w (fun i => a (chunk * 8 + i.val)) (fun i => b (chunk * 8 + i.val))
      (if chunk = 0 then live else true) acc) seed

theorem dot_coverage (k : Nat) (hk : 8 ∣ k) (j : Fin k) :
    j.val / 8 < k / 8 ∧ j.val / 8 * 8 + j.val % 8 = j.val := by
  have hj := j.isLt
  have hm := Nat.mod_eq_zero_of_dvd hk
  omega

abbrev Tensor (α : Type) := Nat → Nat → Nat → α
abbrev Rows (α : Type) := Nat → Nat → α

theorem groups_pos (c : Shape) : 0 < c.groups := Nat.mul_pos c.batches_pos c.heads_pos

def coord (c : Shape) (h i e : Nat) : Coord c :=
  (⟨h % c.groups,Nat.mod_lt _ (groups_pos c)⟩,
   ⟨i % c.rows,Nat.mod_lt _ c.rows_pos⟩,⟨e % c.dim,Nat.mod_lt _ c.dim_pos⟩)
def rowCoord (c : Shape) (h i : Nat) : Fin c.groups × Fin c.rows :=
  (⟨h % c.groups,Nat.mod_lt _ (groups_pos c)⟩,⟨i % c.rows,Nat.mod_lt _ c.rows_pos⟩)

/-- Total logical views repeat outside their extent. Source accesses are all
in bounds by Grid; this convention never adds native loads or source cases. -/
def load (c : Shape) (s : Memory α) (r : Buffer) : Tensor α :=
  fun h i e => s r (offset c (coord c h i e))
def loadRow (c : Shape) (s : Memory α) (r : Buffer) : Rows α :=
  fun h i => s r (rowAddr c (rowCoord c h i))

structure Input (α : Type) where
  q : Tensor α
  k : Tensor α
  v : Tensor α
  g : Tensor α

def input (c : Shape) (s : Memory α) : Input α :=
  ⟨load c s .q,load c s .k,load c s .v,load c s .g⟩

structure ForwardAcc (α : Type) where
  maximum : Fin 4 → α
  denominator : Fin 4 → α
  output : α

def forwardInitial (p : Ops α) : ForwardAcc α :=
  ⟨fun _ => p.negInf,fun _ => p.zero,p.zero⟩

/-- Two source dot sites, scalar order, live PV seed, and four reduction
replicas are retained. Normalized P is half-rounded before FP32 widening. -/
def forwardStep (c : Shape) (p : Ops α) (x : Input α) (scale : α)
    (head row col : Nat) (a : ForwardAcc α) (block : Nat) : ForwardAcc α :=
  let score := fun j => if block * 128 + j ≤ row then
      p.mul scale (dot p .n128 c.dim (x.q head row) (x.k head (block * 128 + j)) false p.zero)
    else p.negInf
  let mx := fun o => p.max ((Reduction.plan o).eval p.max score) (a.maximum o)
  let alpha := fun o => exp p (p.sub (a.maximum o) (mx o))
  let prob := fun j => exp p (p.sub (score j) (mx (Reduction.owner j)))
  let den := fun o => p.fma (a.denominator o) (alpha o) ((Reduction.plan o).eval p.add prob)
  let rcp := fun o => p.div p.one (den o)
  let oldMass := fun o => p.mul (a.denominator o) (alpha o)
  let o := Reduction.owner col
  let acc := p.mul a.output (p.mul (oldMass o) (rcp o))
  let normalized := fun j => half p (p.mul (prob j) (rcp (Reduction.owner j)))
  ⟨mx,den,dot p (width c.dim) 128 normalized (fun j => x.v head (block * 128 + j) col) true acc⟩

def forwardAcc (c : Shape) (p : Ops α) (x : Input α) (scale : α) (h i e : Nat) : ForwardAcc α :=
  (List.range (i / 128 + 1)).foldl (forwardStep c p x scale h i e) (forwardInitial p)

structure Saved (α : Type) where
  y : Tensor α
  l : Rows α
  m : Rows α

def forwardReference (c : Shape) (p : Ops α) (owners : Owners) (x : Input α) (scale : α) : Saved α :=
  ⟨fun h i e => half p (forwardAcc c p x scale (h % c.groups) (i % c.rows) (e % c.dim)).output,
   fun h i => (forwardAcc c p x scale (h % c.groups) (i % c.rows) 0).denominator (owners.l (h % c.groups) (i % c.rows)),
   fun h i => (forwardAcc c p x scale (h % c.groups) (i % c.rows) 0).maximum (owners.m (h % c.groups) (i % c.rows))⟩

/-- In the native FP32 preprocess layout, every lane has four adjacent
channels. Component 1 seeds the product; FMA then incorporates 0,2,3. -/
def preLocal (p : Ops α) (y g : Nat → α) (l : α) (owner : Nat) : α :=
  let d := fun j => p.div (g (owner * 4 + j)) l
  p.fma (d 3) (y (owner * 4 + 3))
    (p.fma (d 2) (y (owner * 4 + 2))
      (p.fma (d 0) (y (owner * 4)) (p.mul (d 1) (y (owner * 4 + 1)))))

def shuffleAdd (p : Ops α) : List Nat → (Nat → α) → Nat → α
  | [],f,o => f o
  | mask :: rest,f,o => shuffleAdd p rest (fun j => p.add (f j) (f (j ^^^ mask))) o

def preMasks (dim : Nat) : List Nat :=
  if dim = 16 then [2,1] else if dim = 32 then [4,2,1]
  else if dim = 64 then [8,4,2,1] else [16,8,4,2,1]

structure Preprocessed (α : Type) where
  newDO : Tensor α
  delta : Rows α

def preprocessReference (c : Shape) (p : Ops α) (owners : Owners)
    (g : Tensor α) (s : Saved α) : Preprocessed α :=
  ⟨fun h i e => p.div (g (h % c.groups) (i % c.rows) (e % c.dim)) (s.l (h % c.groups) (i % c.rows)),
   fun h i => shuffleAdd p (preMasks c.dim)
     (preLocal p (s.y (h % c.groups) (i % c.rows)) (g (h % c.groups) (i % c.rows))
       (s.l (h % c.groups) (i % c.rows)))
     ((owners.delta (h % c.groups) (i % c.rows)) % (c.dim / 4))⟩

structure BackwardInput (α : Type) where
  q : Tensor α
  k : Tensor α
  v : Tensor α
  m : Rows α
  newDO : Tensor α
  delta : Rows α

def backwardInput (c : Shape) (s : Memory α) : BackwardInput α :=
  ⟨load c s .q,load c s .k,load c s .v,loadRow c s .m,
   load c s .newDO,loadRow c s .delta⟩

def bwdP (c : Shape) (p : Ops α) (x : BackwardInput α) (scale : α) (h i j : Nat) : α :=
  let score := if j ≤ i then dot p .n128 c.dim (x.q h i) (x.k h j) false p.zero else p.negInf
  exp p (p.fma scale score (p.neg (x.m h i)))

def bwdDS (c : Shape) (p : Ops α) (x : BackwardInput α) (scale : α) (h i j : Nat) : α :=
  let probability := bwdP c p x scale h i j
  let dp := dot p .n128 c.dim (x.newDO h i) (x.v h j) true (p.neg (x.delta h i))
  p.mul scale (p.mul probability dp)

/-- One repeated source DQ read/add/write, with its actual current value as
WGMMA accumulator seed. Earlier key blocks are never collapsed into a real sum. -/
def dqUpdate (c : Shape) (p : Ops α) (x : BackwardInput α) (scale : α)
    (block h i e : Nat) (current : α) : α :=
  if block ≤ i / 128 then
    dot p (width c.dim) 128 (fun j => half p (bwdDS c p x scale h i (block * 128 + j)))
      (fun j => x.k h (block * 128 + j) e) true current
  else current

def dqReference (c : Shape) (p : Ops α) (x : BackwardInput α) (scale : α)
    (blocks h i e : Nat) : α :=
  (List.range blocks).foldl (fun acc block => dqUpdate c p x scale block h i e acc) p.zero

def dkUpdate (c : Shape) (p : Ops α) (x : BackwardInput α) (scale : α)
    (h j e queryBlock : Nat) (current : α) : α :=
  dot p (width c.dim) 128 (fun r => half p (bwdDS c p x scale h (queryBlock * 128 + r) j))
    (fun r => x.q h (queryBlock * 128 + r) e) true current

def dvUpdate (c : Shape) (p : Ops α) (x : BackwardInput α) (scale : α)
    (h j e queryBlock : Nat) (current : α) : α :=
  dot p (width c.dim) 128 (fun r => half p (bwdP c p x scale h (queryBlock * 128 + r) j))
    (fun r => x.newDO h (queryBlock * 128 + r) e) true current

def dkReference (c : Shape) (p : Ops α) (x : BackwardInput α) (scale : α) (h j e : Nat) : α :=
  half p ((List.range (c.blocks - j / 128)).foldl
    (fun acc t => dkUpdate c p x scale h j e (j / 128 + t) acc) p.zero)
def dvReference (c : Shape) (p : Ops α) (x : BackwardInput α) (scale : α) (h j e : Nat) : α :=
  half p ((List.range (c.blocks - j / 128)).foldl
    (fun acc t => dvUpdate c p x scale h j e (j / 128 + t) acc) p.zero)

/-- Independent source-precision reference: all saved and preprocessing data
are constructed from initial Q/K/V/G and the actual scale word. -/
def wholeBackwardInput (c : Shape) (p : Ops α) (owners : Owners) (x : Input α) (scale : α) : BackwardInput α :=
  let s := forwardReference c p owners x scale
  let pre := preprocessReference c p owners x.g s
  ⟨x.q,x.k,x.v,s.m,pre.newDO,pre.delta⟩

#axiomsClean dot_coverage
#specNonCircular forwardReference avoiding [Memory.scatter]
#specNonCircular wholeBackwardInput avoiding [Memory.scatter]
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
