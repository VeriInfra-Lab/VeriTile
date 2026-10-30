import bench.tritonbench_g.kcache_copy_triton.Kernels
import bench.tritonbench_g.kcache_copy_triton.Proofs.Model
import VeriTile.Triton.Semantics.State

namespace VeriTile.Bench.Repairs.KcacheCopyTriton
open VeriTile.Triton
noncomputable section

def Ranks (m : Inputs) : Prop := ((m .key).shape.length=3 ∨ (m .key).shape.length=4) ∧
  (m .lengths).shape.length=1 ∧ (m .table).shape.length=2

def Types (m : Inputs) : Prop := ∀ r,(m r).dtype=
  if r=.lengths ∨ r=.table then .other "int32" else .floating .fp32

def Shapes (m : Inputs) (p : Config) : Prop := p.count%p.n=0 ∧
  (m .lengths).shape=[p.count/p.n] ∧ (m .table).shape=[p.count/p.n,p.columns]
def CacheShape (m : Inputs) (p : Config) : Prop :=
  if p.newLayout then (m .cache).shape=[p.blocks,p.heads,p.splits,p.blockSize,p.width] ∧ p.splits*p.width=p.dims
  else (m .cache).shape=[p.blocks,p.heads,p.blockSize,p.dims]
def Geometry (p : Config) : Prop := 0<p.heads ∧ 0<p.dims ∧ 0<p.blockSize ∧ ∃ e,p.width=2^e

def Bounds (p : Config) (l : Lengths) (t : Tables) : Prop :=
  (∀ s,s<p.count/p.n → p.n≤(l s).toInt ∧ (l s).toInt≤p.blockSize*p.columns) ∧
  (∀ token,token<p.count → 0≤(t (token/p.n) (column p l token)).toInt ∧
    (t (token/p.n) (column p l token)).toInt<p.blocks)
def Unique (p : Config) (l : Lengths) (t : Tables) : Prop := ∀ s u,s<p.count → u<p.count →
  blockId p l t s=blockId p l t u → slot p l s=slot p l u → s=u

def Separate (m : Inputs) : Prop := ∀ r s,r≠s →
  base m r+4*(m r).numel≤base m s ∨ base m s+4*(m s).numel≤base m r

structure InputContract (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) : Prop where
  nBound : 0<n ∧ n<2^31
  layout : ∀ r,(m r).layout=.strided
  dtype : Types m
  cuda : ∀ r,(m r).device.kind="cuda"
  device : ∀ r,(m r).device=(m .key).device
  rank : Ranks m
  contiguous : ∀ r,(m r).isContiguous=Bool.true
  noNeg : ∀ r,(m r).isNeg=Bool.false
  shapes : Shapes m (config m n b)
  cacheShape : CacheShape m (config m n b)
  geometry : Geometry (config m n b)
  grid : (config m n b).count<2^31 ∧ (config m n b).heads<2^16 ∧ (config m n b).splits<2^16 ∧
    (config m n b).count*(config m n b).heads*(config m n b).splits<2^31
  extent : (∀ r,(m r).numel<2^63) ∧ (config m n b).blockSize<2^63 ∧ (config m n b).columns<2^63
  tile : (config m n b).width≤2^20
  alignment : ∀ r,0<(m r).numel → base m r%4=0
  separate : Separate m
  bounds : Bounds (config m n b) l t
  unique : Unique (config m n b) l t

inductive InputError where
  | tensors | n | selector | layout | dtype | cuda | device | rank | contiguous | negative
  | shapes | cacheShape | geometry | grid | extent | tile | alignment | overlap | bounds | unique
  deriving DecidableEq, Repr
open Classical in
def checkInputs (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables) : Except InputError Unit :=
  let p := config m n b
  if ¬(0<n ∧ n<2^31) then .error .n
  else if ¬(∀ r,(m r).layout=.strided) then .error .layout
  else if ¬Types m then .error .dtype
  else if ¬(∀ r,(m r).device.kind="cuda") then .error .cuda
  else if ¬(∀ r,(m r).device=(m .key).device) then .error .device
  else if ¬Ranks m then .error .rank
  else if ¬(∀ r,(m r).isContiguous=Bool.true) then .error .contiguous
  else if ¬(∀ r,(m r).isNeg=Bool.false) then .error .negative
  else if ¬Shapes m p then .error .shapes
  else if ¬CacheShape m p then .error .cacheShape
  else if ¬Geometry p then .error .geometry
  else if ¬(p.count<2^31 ∧ p.heads<2^16 ∧ p.splits<2^16 ∧ p.count*p.heads*p.splits<2^31) then .error .grid
  else if ¬((∀ r,(m r).numel<2^63) ∧ p.blockSize<2^63 ∧ p.columns<2^63) then .error .extent
  else if ¬p.width≤2^20 then .error .tile
  else if ¬(∀ r,0<(m r).numel → base m r%4=0) then .error .alignment
  else if ¬Separate m then .error .overlap
  else if ¬Bounds p l t then .error .bounds
  else if ¬Unique p l t then .error .unique
  else .ok ()

open Classical in
def checkArguments (m : Option Inputs) (n : Option Int) (b : Option Bool) (l : Lengths) (t : Tables) : Except InputError Unit :=
  match m,n,b with
  | none,_,_ => .error .tensors
  | some _,none,_ => .error .n
  | some m,some n,b => if ¬(0<n ∧ n<2^31) then .error .n else
    match b with | none => .error .selector | some b => checkInputs m n.toNat b l t

def Admitted (m : Option Inputs) (n : Option Int) (b : Option Bool) (l : Lengths) (t : Tables) : Prop :=
  match m,n,b with | some m,some n,some b => 0<n ∧ InputContract m n.toNat b l t | _,_,_ => False

structure TensorBinding (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (decode : α → Word) (heap : Memory α) : Prop where
  capacity : ∀ r,base m r+4*(m r).numel≤2^64
  lengths : ∀ s,s<(config m n b).count/n → decode (heap (address m .lengths s))=l s
  table : ∀ s c,s<(config m n b).count/n → c<(config m n b).columns →
    decode (heap (address m .table (s*(config m n b).columns+c)))=t s c

/-- The exact contiguous strides selected by the Python assertions/launch. -/
def parameters (p : Config) : Source.Parameters :=
  ⟨"key","cache",⟨"table"⟩,⟨"lengths"⟩,p.n,p.heads*p.dims,p.dims,1,
    p.heads*p.blockSize*p.dims,p.blockSize*p.dims,
    if p.newLayout then p.blockSize*p.width else 0,p.width,p.columns,1,p.blockSize,p.width⟩

def programTile (ids : Nat → Nat) : Tile := ⟨ids 0,ids 1,ids 2⟩

/-- Independent tensor formula for every component handled by the selected
program. Universally quantified program coordinates cover all three grid axes. -/
def KernelCopy (p : Config) (l : Lengths) (t : Tables) (ids : Nat → Nat)
    (before after : Nat → ℝ) : Prop :=
  ∀ c,c<p.width → after (outputIndex p l t (ids 0) (ids 1) (ids 2*p.width+c)) =
    before (inputIndex p (ids 0) (ids 1) (ids 2*p.width+c))

structure KernelMetadataBinding (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (s : BlockState) : Prop where
  lengths : ∀ q,q<(config m n b).count/n → s.readMemValue .int "lengths" q=(l q).toInt
  table : ∀ q c,q<(config m n b).count/n → c<(config m n b).columns →
    s.readMemValue .int "table" (q*(config m n b).columns+c)=(t q c).toInt

/-- Initial physical-to-logical interpretation at the actual data pointers.
The Real payload is arbitrary; only INT32 metadata is decoded. -/
def logicalState (m : Inputs) (metadata : Memory Word) (payload : Nat → ℝ)
    (scratch : BlockState) : BlockState :=
  { scratch with mem := fun r i =>
      if r="lengths" then MemCell.of .int (metadata (address m .lengths i)).toInt
      else if r="table" then MemCell.of .int (metadata (address m .table i)).toInt
      else if r="key" then MemCell.real (payload (address m .key i))
      else if r="cache" then MemCell.real (payload (address m .cache i))
      else scratch.mem r i }

end
end VeriTile.Bench.Repairs.KcacheCopyTriton
