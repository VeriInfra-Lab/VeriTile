import bench.repairs.tritonbench_g.kv_cache_copy.Kernels
import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Model
import VeriTile.Triton.Semantics.State

namespace VeriTile.Bench.Repairs.KvCacheCopy
noncomputable section

def InputRank (shape : List Nat) : Prop := shape.length=3 ∨ (shape.length=4 ∧ shape.getD 1 0=1)
def Ranks (m : Inputs) : Prop := InputRank (m .key).shape ∧ InputRank (m .value).shape

def Types (m : Inputs) : Prop := ∀ r, (m r).dtype=
  if r=.lengths ∨ r=.table then .other "int32" else .floating .fp32

def Shapes (m : Inputs) (p : Config) : Prop :=
  normalized (m .value).shape=normalized (m .key).shape ∧
  (m .valueCache).shape=[p.blocks,p.heads,p.blockSize,p.dims] ∧
  (m .lengths).shape=[p.count] ∧ (m .table).shape=[p.count,p.columns]
def KeyShape (m : Inputs) (p : Config) : Prop :=
  if p.newLayout then (m .keyCache).shape=[p.blocks,p.heads,p.splits,p.blockSize,p.width] ∧ p.splits*p.width=p.dims
  else (m .keyCache).shape=(m .valueCache).shape

def Geometry (p : Config) : Prop := 0<p.heads ∧ 0<p.dims ∧ 0<p.blockSize ∧ ∃ e,p.width=2^e

def Bounds (p : Config) (lengths : Lengths) (table : Tables) : Prop := ∀ s,s<p.count →
  1≤(lengths s).toInt ∧ (lengths s).toInt≤p.blockSize*p.columns ∧
  0≤(table s (column p lengths s)).toInt ∧ (table s (column p lengths s)).toInt<p.blocks

def Unique (p : Config) (lengths : Lengths) (table : Tables) : Prop := ∀ s t,
  s<p.count → t<p.count → blockId p lengths table s=blockId p lengths table t →
  slot p lengths s=slot p lengths t → s=t

def Separate (m : Inputs) : Prop := ∀ r s,r≠s →
  base m r+4*(m r).numel≤base m s ∨ base m s+4*(m s).numel≤base m r

structure InputContract (m : Inputs) (newLayout : Bool) (lengths : Lengths) (table : Tables) : Prop where
  layout : ∀ r,(m r).layout=.strided
  dtype : Types m
  cuda : ∀ r,(m r).device.kind="cuda"
  device : ∀ r,(m r).device=(m .key).device
  rank : Ranks m
  contiguous : ∀ r,(m r).isContiguous=Bool.true
  noNeg : ∀ r,(m r).isNeg=Bool.false
  shapes : Shapes m (config m newLayout)
  keyShape : KeyShape m (config m newLayout)
  geometry : Geometry (config m newLayout)
  grid : (config m newLayout).count<2^31 ∧ (config m newLayout).heads<2^16 ∧
    (config m newLayout).count*(config m newLayout).heads<2^31
  extent : (∀ r,(m r).numel<2^31) ∧ (config m newLayout).dims<2^31 ∧
    (config m newLayout).blockSize<2^31 ∧ (config m newLayout).columns<2^31
  tile : (config m newLayout).width≤2^20
  alignment : ∀ r,0<(m r).numel → base m r%4=0
  separate : Separate m
  bounds : Bounds (config m newLayout) lengths table
  unique : Unique (config m newLayout) lengths table

inductive InputError where
  | tensors | selector | layout | dtype | cuda | device | rank | contiguous | negative
  | shapes | keyShape | geometry | grid | extent | tile | alignment | overlap | bounds | unique
  deriving DecidableEq, Repr
open Classical in
def checkInputs (m : Inputs) (newLayout : Bool) (lengths : Lengths) (table : Tables) : Except InputError Unit :=
  let p := config m newLayout
  if ¬(∀ r,(m r).layout=.strided) then .error .layout
  else if ¬Types m then .error .dtype
  else if ¬(∀ r,(m r).device.kind="cuda") then .error .cuda
  else if ¬(∀ r,(m r).device=(m .key).device) then .error .device
  else if ¬Ranks m then .error .rank
  else if ¬(∀ r,(m r).isContiguous=Bool.true) then .error .contiguous
  else if ¬(∀ r,(m r).isNeg=Bool.false) then .error .negative
  else if ¬Shapes m p then .error .shapes
  else if ¬KeyShape m p then .error .keyShape
  else if ¬Geometry p then .error .geometry
  else if ¬(p.count<2^31 ∧ p.heads<2^16 ∧ p.count*p.heads<2^31) then .error .grid
  else if ¬((∀ r,(m r).numel<2^31) ∧ p.dims<2^31 ∧ p.blockSize<2^31 ∧ p.columns<2^31) then .error .extent
  else if ¬p.width≤2^20 then .error .tile
  else if ¬(∀ r,0<(m r).numel → base m r%4=0) then .error .alignment
  else if ¬Separate m then .error .overlap
  else if ¬Bounds p lengths table then .error .bounds
  else if ¬Unique p lengths table then .error .unique
  else .ok ()

def checkArguments (m : Option Inputs) (selector : Option Bool) (lengths : Lengths) (table : Tables) : Except InputError Unit :=
  match m,selector with
  | none,_ => .error .tensors
  | some _,none => .error .selector
  | some m,some b => checkInputs m b lengths table

def Admitted (m : Option Inputs) (selector : Option Bool) (lengths : Lengths) (table : Tables) : Prop :=
  match m,selector with | some m,some b => InputContract m b lengths table | _,_ => False

/-- Only initial readback and the ordinary address-space capacity of direct buffers. -/
structure TensorBinding (m : Inputs) (newLayout : Bool) (lengths : Lengths) (table : Tables)
    (decode : α → Word) (heap : Memory α) : Prop where
  capacity : ∀ r,base m r+4*(m r).numel≤2^64
  lengths : ∀ s,s<(config m newLayout).count → decode (heap (address m .lengths s))=lengths s
  table : ∀ s c,s<(config m newLayout).count → c<(config m newLayout).columns →
    decode (heap (address m .table (s*(config m newLayout).columns+c)))=table s c


open VeriTile.Triton

/-- Exact contiguous strides passed by the checked original launch. -/
def parameters (p : Config) : Source.Parameters :=
  ⟨p.heads*p.dims,p.dims,1,p.heads*p.dims,p.dims,1,
    p.heads*p.blockSize*p.dims,p.blockSize*p.dims,
    if p.newLayout then p.blockSize*p.width else 0,p.width,
    p.heads*p.blockSize*p.dims,p.blockSize*p.dims,p.dims,1,
    p.columns,1,p.blockSize,p.dims,p.width⟩

def inputRegion (key : Bool) : RegionName := if key then "key" else "value"
def outputRegion (key : Bool) : RegionName := if key then "keyCache" else "valueCache"
def ValidProgram (p : Config) (ids : Nat → Nat) : Prop := ids 0<p.count ∧ ids 1<p.heads

/-- Independent complete per-program K/V formula, including every split. -/
def KernelCopy (p : Config) (l : Lengths) (t : Tables) (ids : Nat → Nat)
    (before after : RegionName → Nat → ℝ) : Prop :=
  ∀ key d,d<p.dims →
    after (outputRegion key) (outputIndex p l t key (ids 0) (ids 1) d)=
      before (inputRegion key) (inputIndex p (ids 0) (ids 1) d)

structure KernelMetadataBinding (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (s : BlockState) : Prop where
  lengths : ∀ q,q<(config m b).count → s.readMemValue .int "lengths" q=(l q).toInt
  table : ∀ q c,q<(config m b).count → c<(config m b).columns →
    s.readMemValue .int "table" (q*(config m b).columns+c)=(t q c).toInt

/-- Initial payload and metadata at actual Tensor byte addresses, with no output premise. -/
def logicalState (m : Inputs) (metadata : Memory Word) (payload : Nat → ℝ)
    (scratch : BlockState) : BlockState :=
  { scratch with mem := fun r i =>
      if r="lengths" then MemCell.of .int (metadata (address m .lengths i)).toInt
      else if r="table" then MemCell.of .int (metadata (address m .table i)).toInt
      else if r="key" then MemCell.real (payload (address m .key i))
      else if r="value" then MemCell.real (payload (address m .value i))
      else if r="keyCache" then MemCell.real (payload (address m .keyCache i))
      else if r="valueCache" then MemCell.real (payload (address m .valueCache i))
      else scratch.mem r i }

end
end VeriTile.Bench.Repairs.KvCacheCopy
