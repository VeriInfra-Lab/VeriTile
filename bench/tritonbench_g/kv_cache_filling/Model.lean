import Mathlib
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.KvCacheFilling

inductive Policy where | copy | int4 | int8
  deriving DecidableEq, Repr

structure Request where
  batches : Nat
  tokens : Nat
  heads : Nat
  pages : Nat
  block : Nat
  dk : Nat
  dv : Nat
  maxQ : Nat
  tableWidth : Nat
  policy : Policy
  start : Nat → Nat
  count : Nat → Nat
  total : Nat → Nat
  page : Nat → Nat → Nat

def Request.factor (r : Request) : Nat := if r.policy = .int4 then 2 else 1
def Request.ck (r : Request) : Nat := r.dk / r.factor
def Request.cv (r : Request) : Nat := r.dv / r.factor
def Request.history (r : Request) (b : Nat) : Nat := r.total b - r.count b

structure Job where
  batch : Nat
  token : Nat
  head : Nat
  deriving DecidableEq, Repr

def jobs (r : Request) : List Job :=
  (List.range r.batches).flatMap fun b =>
    (List.range (r.count b)).flatMap fun t =>
      (List.range r.heads).map fun h => ⟨b,t,h⟩

def sourceRow (r : Request) (j : Job) : Nat :=
  (r.start j.batch + j.token) * r.heads + j.head

def destinationToken (r : Request) (b t : Nat) : Nat :=
  r.page b ((r.history b + t) / r.block) * r.block + (r.history b + t) % r.block

def destinationRow (r : Request) (j : Job) : Nat :=
  destinationToken r j.batch j.token * r.heads + j.head

/-- The compiler-selected ordered min/max reduction tree. Leaves name actual
source coordinates, including repeated first coordinates in padding. -/
inductive Tree where
  | leaf : Nat → Tree
  | node : Tree → Tree → Tree
  deriving Repr

def Tree.leaves : Tree → List Nat
  | .leaf i => [i]
  | .node a b => a.leaves ++ b.leaves

def Tree.eval {α : Type} (op : α → α → α) (x : Nat → α) : Tree → α
  | .leaf i => x i
  | .node a b => op (a.eval op x) (b.eval op x)

structure Plan (width : Nat) where
  low : Tree
  high : Tree
  low_valid : ∀ i ∈ low.leaves, i < width
  high_valid : ∀ i ∈ high.leaves, i < width
  low_covers : ∀ i, i < width → i ∈ low.leaves
  high_covers : ∀ i, i < width → i ∈ high.leaves

/-- Source-level primitive ports. Their order is fixed by quantize below.
Division includes the selected constant/runtime division implementation;
no split-rounding/FMA equality or whole-reduction law is assumed. -/
structure Operations (α β : Type) where
  minimum : α → α → α
  maximum : α → α → α
  sub : α → α → α
  div : α → α → α
  add : α → α → α
  neg : α → α
  equal : α → α → Bool
  literal : Nat → α
  half : α
  uint8 : α → β
  pack : β → β → β

inductive Cell (α β : Type) where
  | fp32 : α → Cell α β
  | uint8 : β → Cell α β
  deriving DecidableEq

structure Quantized (α β : Type) where
  code : Nat → β
  scale : α
  zero : α

def quantize {α β : Type} (ops : Operations α β) (levels : Nat)
    (x : Nat → α) (low high : Tree) : Quantized α β :=
  let lo := low.eval ops.minimum x
  let hi := high.eval ops.maximum x
  let raw := ops.div (ops.sub hi lo) (ops.literal levels)
  let scale := if ops.equal hi lo then ops.literal 1 else raw
  let zero := ops.div (ops.neg lo) scale
  ⟨fun i => ops.uint8 (ops.add (ops.add (ops.div (x i) scale) zero) ops.half), scale, zero⟩

structure RowResult (α β : Type) where
  value : Nat → Cell α β
  scale : α
  zero : α

def rowKernel {α β : Type} (ops : Operations α β) (policy : Policy)
    (packedWidth : Nat) (x : Nat → α) (low high : Tree) : RowResult α β :=
  match policy with
  | .copy => ⟨fun i => .fp32 (x i), ops.literal 0, ops.literal 0⟩
  | .int8 =>
      let q := quantize ops 255 x low high
      ⟨fun i => .uint8 (q.code i), q.scale, q.zero⟩
  | .int4 =>
      let q := quantize ops 15 x low high
      ⟨fun i => .uint8 (ops.pack (q.code i) (q.code (i + packedWidth))), q.scale, q.zero⟩

inductive Role where | key | value | keyMeta | valueMeta
  deriving DecidableEq, Repr

structure Address where
  role : Role
  row : Nat
  column : Nat
  deriving DecidableEq, Repr

structure Store (ν : Type) where
  address : Address
  value : ν

def Request.width (r : Request) : Role → Nat
  | .key => r.ck
  | .value => r.cv
  | .keyMeta | .valueMeta => 2

def rowAddresses (r : Request) (j : Job) : List Address :=
  let row := destinationRow r j
  (List.range r.ck).map (fun i => ⟨.key,row,i⟩) ++
  (List.range r.cv).map (fun i => ⟨.value,row,i⟩) ++
  (if r.policy = .copy then [] else
    [⟨.keyMeta,row,0⟩,⟨.keyMeta,row,1⟩,⟨.valueMeta,row,0⟩,⟨.valueMeta,row,1⟩])

def rowValue {α β : Type} (key value : RowResult α β) (a : Address) : Cell α β :=
  match a.role with
  | .key => key.value a.column
  | .value => value.value a.column
  | .keyMeta => .fp32 (if a.column = 0 then key.scale else key.zero)
  | .valueMeta => .fp32 (if a.column = 0 then value.scale else value.zero)

/-- Actual row loads use the same SRC row for K and V. Metadata stores consume
the scale/zero just produced by those row quantizers, never an independent input. -/
def taskValue {α β : Type} (r : Request) (ops : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (j : Job) : Address → Cell α β :=
  rowValue (rowKernel ops r.policy r.ck (k (sourceRow r j)) kp.low kp.high)
    (rowKernel ops r.policy r.cv (v (sourceRow r j)) vp.low vp.high)

def taskStores {α β : Type} (r : Request) (ops : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (j : Job) : List (Store (Cell α β)) :=
  (rowAddresses r j).map fun a => ⟨a, taskValue r ops kp vp k v j a⟩

def stores {α β : Type} (r : Request) (ops : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) : List (Store (Cell α β)) :=
  (jobs r).flatMap (taskStores r ops kp vp k v)

def runStores {ν : Type} : List (Store ν) → (Address → ν) → Address → ν
  | [], memory => memory
  | s :: rest, memory => runStores rest (Function.update memory s.address s.value)

def execute {α β : Type} (r : Request) (ops : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (memory : Address → Cell α β) :=
  runStores (stores r ops kp vp k v) memory

def Tree.substitute (f : Nat → Tree) : Tree → Tree
  | .leaf i => f i
  | .node a b => .node (a.substitute f) (b.substitute f)

def padIndex (n i : Nat) : Nat := if i < n then i else 0

def pairLeaves (n i : Nat) : Tree :=
  .node (.leaf (padIndex n i)) (.leaf (padIndex n i + n))

def maskedLoad {α : Type} (zero : α) (n : Nat) (x : Nat → α) (i : Nat) : α :=
  if i < n then x i else zero

def paddedLoad {α : Type} (zero : α) (n : Nat) (x : Nat → α) (i : Nat) : α :=
  if i < n then maskedLoad zero n x i else x 0

inductive QReg where | input | low | high | span | rawScale | scale | minusLow | zero | quotient | shifted | rounded
  deriving DecidableEq
inductive QExpr where
  | reg : QReg → QExpr
  | literal : Nat → QExpr
  | half : QExpr
  | sub : QExpr → QExpr → QExpr
  | div : QExpr → QExpr → QExpr
  | add : QExpr → QExpr → QExpr
  | neg : QExpr → QExpr
  | selectEqual : QExpr → QExpr → QExpr → QExpr → QExpr

def QExpr.eval {α β : Type} (ops : Operations α β) (regs : QReg → α) : QExpr → α
  | .reg r => regs r
  | .literal n => ops.literal n
  | .half => ops.half
  | .sub a b => ops.sub (a.eval ops regs) (b.eval ops regs)
  | .div a b => ops.div (a.eval ops regs) (b.eval ops regs)
  | .add a b => ops.add (a.eval ops regs) (b.eval ops regs)
  | .neg a => ops.neg (a.eval ops regs)
  | .selectEqual a b t f => if ops.equal (a.eval ops regs) (b.eval ops regs) then t.eval ops regs else f.eval ops regs

def quantSource (levels : Nat) : List (QReg × QExpr) :=
  [(.span,.sub (.reg .high) (.reg .low)),
   (.rawScale,.div (.reg .span) (.literal levels)),
   (.scale,.selectEqual (.reg .high) (.reg .low) (.literal 1) (.reg .rawScale)),
   (.minusLow,.neg (.reg .low)),
   (.zero,.div (.reg .minusLow) (.reg .scale)),
   (.quotient,.div (.reg .input) (.reg .scale)),
   (.shifted,.add (.reg .quotient) (.reg .zero)),
   (.rounded,.add (.reg .shifted) .half)]

def runQuantSource {α β : Type} (ops : Operations α β) (levels : Nat) (x lo hi : α) : QReg → α :=
  (quantSource levels).foldl (fun regs s => Function.update regs s.1 (s.2.eval ops regs))
    (fun r => match r with | .input => x | .low => lo | .high => hi | _ => ops.literal 0)

structure SourceBinding {n : Nat} (p : Plan n) (policy : Policy) (width : Nat) where
  lanes : Nat
  lane_count : lanes=width.nextPowerOfTwo
  covers : width ≤ lanes
  reduction : Plan lanes
  low : p.low = match policy with
    | .copy | .int8 => reduction.low.substitute (fun i => .leaf (padIndex width i))
    | .int4 => reduction.low.substitute (pairLeaves width)
  high : p.high = match policy with
    | .copy | .int8 => reduction.high.substitute (fun i => .leaf (padIndex width i))
    | .int4 => reduction.high.substitute (pairLeaves width)

def sourceQuant {α β : Type} (ops : Operations α β) (levels : Nat) (x : Nat → α) (lo hi : α) : Quantized α β :=
  ⟨fun i => ops.uint8 (runQuantSource ops levels (x i) lo hi .rounded),
    runQuantSource ops levels (x 0) lo hi .scale,
    runQuantSource ops levels (x 0) lo hi .zero⟩

def sourceRowExec {α β : Type} (ops : Operations α β) (policy : Policy) (width : Nat)
    (x : Nat → α) (low high : Tree) : RowResult α β :=
  match policy with
  | .copy => ⟨fun i => .fp32 (maskedLoad (ops.literal 0) width x i),ops.literal 0,ops.literal 0⟩
  | .int8 =>
    let padded := paddedLoad (ops.literal 0) width x
    let q := sourceQuant ops 255 x (low.eval ops.minimum padded) (high.eval ops.maximum padded)
    ⟨fun i => .uint8 (q.code i),q.scale,q.zero⟩
  | .int4 =>
    let first := paddedLoad (ops.literal 0) width x
    let second := paddedLoad (ops.literal 0) width (fun i => x (i+width))
    let q := sourceQuant ops 15 x
      (low.eval ops.minimum (fun i => ops.minimum (first i) (second i)))
      (high.eval ops.maximum (fun i => ops.maximum (first i) (second i)))
    ⟨fun i => .uint8 (ops.pack (q.code i) (q.code (i+width))),q.scale,q.zero⟩

end VeriTile.Bench.Repairs.KvCacheFilling
