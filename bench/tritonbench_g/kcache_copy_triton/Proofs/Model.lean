import bench.tritonbench_g.kv_cache_copy.Proofs.Model
namespace VeriTile.Bench.Repairs.KcacheCopyTriton
noncomputable section
abbrev Word := KvCacheCopy.Word
abbrev Memory := KvCacheCopy.Memory
abbrev Lengths := KvCacheCopy.Lengths
abbrev Tables := KvCacheCopy.Tables
abbrev Tile := KvCacheCopy.Tile
abbrev scatter := @KvCacheCopy.scatter

inductive Role where
  | key | cache | lengths | table
  deriving DecidableEq, Repr
abbrev Inputs := Role → VeriTile.Torch.TensorMetadata

def normalized (shape : List Nat) : List Nat :=
  if shape.length=4 then [shape.getD 0 0*shape.getD 1 0,shape.getD 2 0,shape.getD 3 0] else shape

structure Config extends KvCacheCopy.Config where
  n : Nat

def config (m : Inputs) (n : Nat) (newLayout : Bool) : Config :=
  let shape := normalized (m .key).shape
  ⟨⟨shape.getD 0 0,shape.getD 1 0,shape.getD 2 0,(m .cache).shape.getD 0 0,
    (m .cache).shape.getD (if newLayout then 3 else 2) 0,(m .table).shape.getD 1 0,
    if newLayout then (m .cache).shape.getD 2 0 else 1,
    if newLayout then (m .cache).shape.getD 4 0 else shape.getD 2 0,newLayout⟩,n⟩

def base (m : Inputs) (r : Role) : Nat := (m r).dataPtr.toNat
def address (m : Inputs) (r : Role) (i : Nat) : Nat := base m r+4*i

def past (p : Config) (lengths : Lengths) (token : Nat) : Nat :=
  (lengths (token/p.n)).toInt.toNat-p.n+token%p.n
def column (p : Config) (lengths : Lengths) (token : Nat) : Nat := past p lengths token/p.blockSize
def slot (p : Config) (lengths : Lengths) (token : Nat) : Nat := past p lengths token%p.blockSize
def blockId (p : Config) (lengths : Lengths) (table : Tables) (token : Nat) : Nat :=
  (table (token/p.n) (column p lengths token)).toInt.toNat

def inputIndex (p : Config) (s h d : Nat) : Nat := (s*p.heads+h)*p.dims+d
def outputIndex (p : Config) (lengths : Lengths) (table : Tables) (s h d : Nat) : Nat :=
  KvCacheCopy.cacheIndex p.toConfig p.newLayout (blockId p lengths table s) h (slot p lengths s) d

structure Lane where
  tile : Tile
  lane : Nat
  deriving DecidableEq

def tiles (p : Config) : List Tile := KvCacheCopy.tiles p.toConfig
def tileLanes (p : Config) (t : Tile) : List Lane := (List.range p.width).map fun c => ⟨t,c⟩
def lanes (p : Config) : List Lane := (tiles p).flatMap (tileLanes p)
def component (p : Config) (e : Lane) : Nat := e.tile.split*p.width+e.lane

def sourceAddress (m : Inputs) (p : Config) (e : Lane) : Nat :=
  address m .key (inputIndex p e.tile.sequence e.tile.head (component p e))
def targetAddress (m : Inputs) (p : Config) (lengths : Lengths) (table : Tables) (e : Lane) : Nat :=
  address m .cache (outputIndex p lengths table e.tile.sequence e.tile.head (component p e))

def tile (m : Inputs) (p : Config) (lengths : Lengths) (table : Tables)
    (heap : Memory α) (t : Tile) : Memory α :=
  scatter (tileLanes p t) (targetAddress m p lengths table) (fun e => heap (sourceAddress m p e)) heap

def launch (m : Inputs) (p : Config) (lengths : Lengths) (table : Tables) (heap : Memory α) : Memory α :=
  (tiles p).foldl (tile m p lengths table) heap

def CacheCopy (m : Inputs) (p : Config) (lengths : Lengths) (table : Tables)
    (before after : Memory α) : Prop :=
  (∀ s h d,s<p.count → h<p.heads → d<p.dims →
    after (address m .cache (outputIndex p lengths table s h d))=before (address m .key (inputIndex p s h d))) ∧
  (∀ a,(∀ s h d,s<p.count → h<p.heads → d<p.dims →
    a≠address m .cache (outputIndex p lengths table s h d)) → after a=before a)

namespace Native
abbrev Index := BitVec 64
def i (n : Nat) : Index := BitVec.ofNat 64 n
def ptr (m : Inputs) (role : Role) (offset : Index) : Nat := (i (base m role)+offset*i 4).toNat

def sourceIndex (p : Config) (e : Lane) : Index :=
  i e.tile.sequence*i (p.heads*p.dims)+i e.tile.head*i p.dims+(i e.tile.split*i p.width+i e.lane)
def targetIndex (p : Config) (block past : Index) (e : Lane) : Index :=
  let position := BitVec.ofInt 64 (past.toInt.tmod (p.blockSize:Int))
  block*i (p.heads*p.blockSize*p.dims)+i e.tile.head*i (p.blockSize*p.dims)+
    i e.tile.split*i (if p.newLayout then p.blockSize*p.width else 0)+position*i p.width+i e.lane

def resolve (m : Inputs) (p : Config) (decode : α → Word) (heap : Memory α) (token : Nat) : Index × Index :=
  let seq := token/p.n
  let shift := i token-i p.n*(i seq+i 1)
  let past := BitVec.ofInt 64 (decode (heap (ptr m .lengths (i seq)))).toInt+shift
  let col := BitVec.ofInt 64 (past.toInt.tdiv (p.blockSize:Int))
  let block := BitVec.ofInt 64 (decode (heap (ptr m .table (i seq*i p.columns+col)))).toInt
  (block,past)

def tile (m : Inputs) (p : Config) (decode : α → Word) (heap : Memory α) (t : Tile) : Memory α :=
  let resolved := resolve m p decode heap t.sequence
  scatter (tileLanes p t) (fun e => ptr m .cache (targetIndex p resolved.1 resolved.2 e))
    (fun e => heap (ptr m .key (sourceIndex p e))) heap

def launch (m : Inputs) (p : Config) (decode : α → Word) (heap : Memory α) : Memory α :=
  (tiles p).foldl (tile m p decode) heap
end Native
end
end VeriTile.Bench.Repairs.KcacheCopyTriton
