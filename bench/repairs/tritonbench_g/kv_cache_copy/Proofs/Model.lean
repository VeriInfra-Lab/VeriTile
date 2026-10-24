import VeriTile.Torch.Tensor
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Log

namespace VeriTile.Bench.Repairs.KvCacheCopy
noncomputable section

inductive Role where
  | key | value | keyCache | valueCache | lengths | table
  deriving DecidableEq, Repr
abbrev Inputs := Role → VeriTile.Torch.TensorMetadata
abbrev Word := BitVec 32
abbrev Memory (α : Type) := Nat → α
abbrev Lengths := Nat → Word
abbrev Tables := Nat → Nat → Word

def normalized (shape : List Nat) : List Nat :=
  if shape.length=4 then [shape.getD 0 0,shape.getD 2 0,shape.getD 3 0] else shape
structure Config where
  (count heads dims blocks blockSize columns splits width : Nat)
  newLayout : Bool

def config (m : Inputs) (newLayout : Bool) : Config :=
  let shape := normalized (m .key).shape
  ⟨shape.getD 0 0,shape.getD 1 0,shape.getD 2 0,
    (m .valueCache).shape.getD 0 0,(m .valueCache).shape.getD 2 0,(m .table).shape.getD 1 0,
    if newLayout then (m .keyCache).shape.getD 2 0 else 1,
    if newLayout then (m .keyCache).shape.getD 4 0 else shape.getD 2 0,newLayout⟩

def base (m : Inputs) (r : Role) : Nat := (m r).dataPtr.toNat
def address (m : Inputs) (r : Role) (i : Nat) : Nat := base m r+4*i

def past (lengths : Lengths) (s : Nat) : Nat := (lengths s).toInt.toNat-1
def column (p : Config) (lengths : Lengths) (s : Nat) : Nat := past lengths s/p.blockSize
def slot (p : Config) (lengths : Lengths) (s : Nat) : Nat := past lengths s%p.blockSize
def blockId (p : Config) (lengths : Lengths) (table : Tables) (s : Nat) : Nat :=
  (table s (column p lengths s)).toInt.toNat

def inputIndex (p : Config) (s h d : Nat) : Nat := (s*p.heads+h)*p.dims+d

def cacheIndex (p : Config) (newLayout : Bool) (block h position d : Nat) : Nat :=
  if newLayout then (((block*p.heads+h)*p.splits+d/p.width)*p.blockSize+position)*p.width+d%p.width
  else ((block*p.heads+h)*p.blockSize+position)*p.dims+d

def outputRole (key : Bool) : Role := if key then .keyCache else .valueCache
def inputRole (key : Bool) : Role := if key then .key else .value

def outputIndex (p : Config) (lengths : Lengths) (table : Tables)
    (key : Bool) (s h d : Nat) : Nat :=
  cacheIndex p (key && p.newLayout) (blockId p lengths table s) h (slot p lengths s) d

structure Tile where
  (sequence head split : Nat)
  deriving DecidableEq
structure Lane where
  tile : Tile
  key : Bool
  lane : Nat
  deriving DecidableEq

def tiles (p : Config) : List Tile :=
  (List.range p.count).flatMap fun s => (List.range p.heads).flatMap fun h =>
    (List.range p.splits).map fun q => ⟨s,h,q⟩
def tileLanes (p : Config) (t : Tile) : List Lane :=
  (List.range p.width).map (fun c => ⟨t,true,c⟩) ++ (List.range p.width).map (fun c => ⟨t,false,c⟩)
def lanes (p : Config) : List Lane := (tiles p).flatMap (tileLanes p)
def component (p : Config) (e : Lane) : Nat := e.tile.split*p.width+e.lane

def sourceAddress (m : Inputs) (p : Config) (e : Lane) : Nat :=
  address m (inputRole e.key) (inputIndex p e.tile.sequence e.tile.head (component p e))
def targetAddress (m : Inputs) (p : Config) (lengths : Lengths) (table : Tables) (e : Lane) : Nat :=
  address m (outputRole e.key) (outputIndex p lengths table e.key e.tile.sequence e.tile.head (component p e))

def scatter {ι α : Type} (events : List ι) (key : ι → Nat) (value : ι → α) (heap : Memory α) : Memory α :=
  events.foldl (fun state e => Function.update state (key e) (value e)) heap

/-- Source-order pair of vector stores. Both K and V are loaded before either
store, and a tile captures its incoming payload values before scattering. -/
def tile (m : Inputs) (p : Config) (lengths : Lengths) (table : Tables)
    (heap : Memory α) (t : Tile) : Memory α :=
  scatter (tileLanes p t) (targetAddress m p lengths table)
    (fun e => heap (sourceAddress m p e)) heap

/-- Mutable execution of the complete static split loop and both launch IDs. -/
def launch (m : Inputs) (p : Config) (lengths : Lengths) (table : Tables) (heap : Memory α) : Memory α :=
  (tiles p).foldl (tile m p lengths table) heap

/-- Independent physical scatter target retaining the initial source values. -/
def CacheCopy (m : Inputs) (p : Config) (lengths : Lengths) (table : Tables)
    (before after : Memory α) : Prop :=
  (∀ key s h d, s<p.count → h<p.heads → d<p.dims →
    after (address m (outputRole key) (outputIndex p lengths table key s h d))=
      before (address m (inputRole key) (inputIndex p s h d))) ∧
  (∀ a, (∀ key s h d, s<p.count → h<p.heads → d<p.dims →
    a≠address m (outputRole key) (outputIndex p lengths table key s h d)) → after a=before a)

namespace Native
def i (n : Nat) : Word := BitVec.ofNat 32 n
def ptr (m : Inputs) (role : Role) (offset : Word) : Nat :=
  (BitVec.ofNat 64 (base m role)+BitVec.ofInt 64 offset.toInt*BitVec.ofNat 64 4).toNat

def sourceIndex (p : Config) (e : Lane) : Word :=
  i e.tile.sequence*i (p.heads*p.dims)+i e.tile.head*i p.dims+
    (i e.tile.split*i p.width+i e.lane)
def targetIndex (p : Config) (block past : Word) (e : Lane) : Word :=
  let position := BitVec.ofInt 32 (past.toInt.tmod (p.blockSize:Int))
  if e.key then
    block*i (p.heads*p.blockSize*p.dims)+i e.tile.head*i (p.blockSize*p.dims)+
      i e.tile.split*i (if p.newLayout then p.blockSize*p.width else 0)+position*i p.width+i e.lane
  else
    block*i (p.heads*p.blockSize*p.dims)+i e.tile.head*i (p.blockSize*p.dims)+
      position*i p.dims+(i e.tile.split*i p.width+i e.lane)

def resolve (m : Inputs) (p : Config) (decode : α → Word) (heap : Memory α) (s : Nat) : Word × Word :=
  let past := decode (heap (ptr m .lengths (i s)))-i 1
  let col := BitVec.ofInt 32 (past.toInt.tdiv (p.blockSize:Int))
  let block := decode (heap (ptr m .table (i s*i p.columns+col)))
  (block,past)

/-- Exact per-split loads followed by K stores and V stores, with a prelude
captured once at entry to the original static loop. -/
def tile (m : Inputs) (p : Config) (resolved : Word × Word) (heap : Memory α) (t : Tile) : Memory α :=
  scatter (tileLanes p t) (fun e => ptr m (outputRole e.key) (targetIndex p resolved.1 resolved.2 e))
    (fun e => heap (ptr m (inputRole e.key) (sourceIndex p e))) heap

def programs (p : Config) : List (Nat × Nat) :=
  (List.range p.count).flatMap fun s => (List.range p.heads).map fun h => (s,h)

def program (m : Inputs) (p : Config) (decode : α → Word) (heap : Memory α) (id : Nat × Nat) : Memory α :=
  let resolved := resolve m p decode heap id.1
  ((List.range p.splits).map fun q => (Tile.mk id.1 id.2 q)).foldl (tile m p resolved) heap

def launch (m : Inputs) (p : Config) (decode : α → Word) (heap : Memory α) : Memory α :=
  (programs p).foldl (program m p decode) heap
end Native

end
end VeriTile.Bench.Repairs.KvCacheCopy
