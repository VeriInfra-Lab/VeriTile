import bench.repairs.tritonbench_g.rotary_emb.Proofs.Model
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.RotaryEmbNopad
abbrev Ops := VeriTile.Bench.Repairs.RotaryEmb.Ops
abbrev Region := VeriTile.Bench.Repairs.RotaryEmb.Region
abbrev Memory := VeriTile.Bench.Repairs.RotaryEmb.Memory

/-- The unchanged arithmetic in each Q/K pair; compiler fusion is retained. -/
def sourcePair {α : Type} (p : Ops α) (a b c s : α) : α×α :=
  (p.sub (p.mul a c) (p.mul b s),p.add (p.mul a s) (p.mul b c))

structure Request where
  tokens : Nat
  qHeads : Nat
  kHeads : Nat
  half : Nat
  cosRows : Nat
  cosWidth : Nat
  sinRows : Nat
  sinWidth : Nat
  cached : Bool
  blocks : Nat
  blockSize : Nat
  tableColumns : Nat
  device : Nat
  /-- Actual aligned pointers measured in four-byte words. -/
  qBase : Nat
  kBase : Nat
  cosBase : Nat
  sinBase : Nat
  cacheBase : Nat
  tableBase : Nat
  lengthBase : Nat

def Request.width (r : Request) := 2*r.half
def Request.heads (r : Request) (key : Bool) := if key then r.kHeads else r.qHeads
def Request.groups (r : Request) := r.qHeads/r.kHeads
def Request.blockPairs (r : Request) := 2^Nat.clog 2 r.half
def Request.blockTokens (r : Request) := if r.cached then 1 else 4
def Request.headPrograms (r : Request) := if r.cached then 2^Nat.clog 2 r.qHeads else r.qHeads
def Request.tokenPrograms (r : Request) := (r.tokens+r.blockTokens-1)/r.blockTokens

def Request.dim (r : Request) (pair : Nat) (second : Bool) := pair+(if second then r.half else 0)
def Request.data (r : Request) (key : Bool) (token head dim : Nat) :=
  (if key then r.kBase else r.qBase)+(token*r.heads key+head)*r.width+dim

def Request.coefficient (r : Request) (sine : Bool) (token pair : Nat) :=
  (if sine then r.sinBase else r.cosBase)+token*(if sine then r.sinWidth else r.cosWidth)+pair

/-- A metadata read is a signed int64 load; subsequent subtraction/division
and the table lookup are the actual source expressions. -/
abbrev Indices := Nat → Int

def Request.past (r : Request) (indices : Indices) (token : Nat) : Nat :=
  (indices (r.lengthBase+2*token)-1).toNat

def Request.block (r : Request) (indices : Indices) (token : Nat) : Nat :=
  (indices (r.tableBase+2*(token*r.tableColumns+r.past indices token/r.blockSize))).toNat

def Request.slot (r : Request) (indices : Indices) (token : Nat) := r.past indices token%r.blockSize

def Request.destination (r : Request) (indices : Indices) (key : Bool) (token head dim : Nat) :=
  if r.cached && key then
    r.cacheBase+((r.block indices token*r.kHeads+head)*r.blockSize+r.slot indices token)*r.width+dim
  else r.data key token head dim

structure Work where
  key : Bool
  token : Nat
  head : Nat
  pair : Nat
  deriving DecidableEq,Repr

def Request.Active (r : Request) (w : Work) : Prop :=
  w.token<r.tokens ∧ w.head<r.heads w.key ∧ w.pair<r.half
instance (r : Request) (w : Work) : Decidable (r.Active w) := inferInstanceAs (Decidable (_∧_∧_))

def sourceValues {α : Type} (p : Ops α) (r : Request) (w : Work) (m : Memory α) : α×α :=
  sourcePair p
    (m (.other r.device) (r.data w.key w.token w.head (r.dim w.pair false)))
    (m (.other r.device) (r.data w.key w.token w.head (r.dim w.pair true)))
    (m (.other r.device) (r.coefficient false w.token w.pair))
    (m (.other r.device) (r.coefficient true w.token w.pair))

def sourceStep {α : Type} (p : Ops α) (r : Request) (indices : Indices) (w : Work) (m : Memory α) : Memory α :=
  let value := sourceValues p r w m
  let low := r.destination indices w.key w.token w.head (r.dim w.pair false)
  let high := r.destination indices w.key w.token w.head (r.dim w.pair true)
  let afterLow := fun region a => if region=.other r.device ∧ a=low then value.1 else m region a
  fun region a => if region=.other r.device ∧ a=high then value.2 else afterLow region a

end VeriTile.Bench.Repairs.RotaryEmbNopad
