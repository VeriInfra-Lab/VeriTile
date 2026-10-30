import VeriTile.Triton.Memory.LogicalIO
import bench.tritonbench_g.mixed_sparse_attention.Proofs.MixedSparseAttention_Real_RealBackend
import bench.tritonbench_g.mixed_sparse_attention.Proofs.KernelGrid
import VeriTile.Triton.Float.ArrayProgramIO

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked
noncomputable section
open Classical
variable {α : Type}

/-- Eight named input arrays. Their logical domains remain Nat: the old raw
source theorem places no count/index bound on sparse selections. Only the
finite source-free access footprint below requires storage backing. -/
def inputs (c : Config) : List ArrayIOChannel :=
  [⟨.real,c.q⟩,⟨.real,c.k⟩,⟨.real,c.v⟩,⟨.nat,Region.cast c.seq⟩,
   ⟨.nat,Region.cast c.blocks⟩,⟨.nat,Region.cast c.offsets⟩,
   ⟨.nat,Region.cast c.counts⟩,⟨.nat,Region.cast c.columns⟩]
abbrev Values (α : Type) (c : Config) := ArrayValues α (inputs c)

def query (c : Config) (xs : Values α c) : Nat → α := xs (0 : Fin 8)
def key (c : Config) (xs : Values α c) : Nat → α := xs (1 : Fin 8)
def value (c : Config) (xs : Values α c) : Nat → α := xs (2 : Fin 8)
def length (c : Config) (xs : Values α c) : Nat → Nat := xs (3 : Fin 8)
def blocks (c : Config) (xs : Values α c) : Nat → Nat := xs (4 : Fin 8)
def offsets (c : Config) (xs : Values α c) : Nat → Nat := xs (5 : Fin 8)
def counts (c : Config) (xs : Values α c) : Nat → Nat := xs (6 : Fin 8)
def columns (c : Config) (xs : Values α c) : Nat → Nat := xs (7 : Fin 8)

def slot (c : Config) (p : Program) := p.2.1*c.NR+p.1
def row (c : Config) (p : Program) := p.1*c.BM
def sequence (c : Config) (xs : Values α c) (p : Program) := length c xs (p.2.1/c.H)
def blockCount (c : Config) (xs : Values α c) (p : Program) := blocks c xs (slot c p)
def columnCount (c : Config) (xs : Values α c) (p : Program) := counts c xs (slot c p)
def qbase (c : Config) (p : Program) := p.2.1/c.H*c.sqz+p.2.1%c.H*c.sqh
def kvbase (c : Config) (p : Program) := p.2.1/c.H*c.skz+p.2.1%c.H*c.skh

def selected (c : Config) (xs : Values α c) (p : Program) (ph : Bool) (n : Nat) (j : Fin c.BN) :=
  if ph then
    if n+j.val<columnCount c xs p ∧ n<columnCount c xs p then
      columns c xs (slot c p*c.NV+n+j.val) else 0
  else (if n<blockCount c xs p then offsets c xs (slot c p*c.NS+n) else 0)+j.val

def keyActive (c : Config) (xs : Values α c) (p : Program) (ph : Bool) (n : Nat) (j : Fin c.BN) : Prop :=
  if ph then n+j.val<columnCount c xs p ∧ n<columnCount c xs p
  else selected c xs p Bool.false n j<sequence c xs p ∧ n<blockCount c xs p

def lanes (shape : TileShape) : List (TileIndex shape) := TileShape.allIndices shape

def queryReads (c : Config) (xs : Values α c) (p : Program) : List Nat :=
  ((lanes [c.BM,c.BD]).filter fun i=>decide (row c p+i.1.val<sequence c xs p)).map
    fun i=>qbase c p+(row c p+i.1.val)*c.sqm+i.2.1.val*c.sqk

def sparseReads (c : Config) (xs : Values α c) (p : Program) (ph : Bool) (dimStride tokenStride : Nat) : List Nat :=
  (List.range (if ph then columnCount c xs p else blockCount c xs p)).flatMap fun n=>
    ((lanes [c.BN,c.BD]).filter fun i=>decide (keyActive c xs p ph n i.1)).map
      fun i=>kvbase c p+i.2.1.val*dimStride+selected c xs p ph n i.1*tokenStride

def keyReads (c : Config) (xs : Values α c) (p : Program) : List Nat :=
  sparseReads c xs p Bool.false c.skk c.skn ++ sparseReads c xs p Bool.true c.skk c.skn

def valueReads (c : Config) (xs : Values α c) (p : Program) : List Nat :=
  sparseReads c xs p Bool.false c.svk c.svn ++ sparseReads c xs p Bool.true c.svk c.svn

def reads (c : Config) (i : Fin (inputs c).length) (xs : Values α c) : List Nat :=
  let jobs:=programs (grid c)
  match i.val with
  | 0=>jobs.flatMap (queryReads c xs)
  | 1=>jobs.flatMap (keyReads c xs)
  | 2=>jobs.flatMap (valueReads c xs)
  | 3=>jobs.map fun p=>p.2.1/c.H
  | 4=>jobs.map (slot c)
  | 5=>jobs.flatMap fun p=>(List.range (blockCount c xs p)).map fun n=>slot c p*c.NS+n
  | 6=>jobs.map (slot c)
  | _=>jobs.flatMap fun p=>(List.range (columnCount c xs p)).map fun n=>slot c p*c.NV+n

def outputs (c : Config) : List IOChannel := [⟨.real,[c.Z*c.H,c.T,c.BD],c.out⟩]

def coordinate (c : Config) (i : TileIndex [c.Z*c.H,c.T,c.BD]) : Raw.Coord c :=
  (i.1,i.2.1,i.2.2.1)
def active (c : Config) (xs : Values α c) (i : TileIndex [c.Z*c.H,c.T,c.BD]) : Prop :=
  i.2.1.val<length c xs (i.1.val/c.H)
def address (c : Config) (i : TileIndex [c.Z*c.H,c.T,c.BD]) : Nat :=
  (i.1.val*c.T+i.2.1.val)*c.BD+i.2.2.1.val

/-- Fresh registers and program IDs, unchanged memory and undef, precisely as
Whole.start. The source is the complete original parsed typed kernel. -/
def invocation (M : Algebra α) (D : Dot α) (c : Config) (p : Program) : Invocation α where
  kernel := kernel c
  backend := backend M D
  pids := ids p
  numPids := some (dimensions (grid c))
  clearRegisters := Bool.true

def ioWith {β : Type} (M : Algebra β) (D : Dot β) (c : Config) (work : List Program)
    (lift : Values α c → Values β c) (encode : α → β) (observe : β → α)
    (writeOutput : (o : Fin (outputs c).length) → Values α c → TileIndex (outputs c)[o].shape → Nat)
    (activeOutput : (o : Fin (outputs c).length) → Values α c → TileIndex (outputs c)[o].shape → Prop) :
    ArrayProgramIO α β where
  inputs := inputs c
  outputs := outputs c
  buffers := [c.q,c.k,c.v,Region.cast c.seq,Region.cast c.blocks,Region.cast c.offsets,
    Region.cast c.counts,Region.cast c.columns,c.out]
  reads := fun i xs => reads c i (lift xs)
  write := writeOutput
  outputMask := activeOutput
  scratch := []
  scratchWrite := fun k=>Fin.elim0 k
  scratchMask := fun k=>Fin.elim0 k
  schedule := fun _=>work.map (invocation M D c)
  encode := encode
  observe := fun _ _=>observe

def io (M : Algebra α) (D : Dot α) (c : Config) (work : List Program) : ArrayProgramIO α α :=
  ioWith M D c work id id id
    (fun | ⟨0,_⟩,_,i=>address c i)
    (fun | ⟨0,_⟩,xs,i=>active c xs i)

/-- Exactly the existing raw sequence-length condition; no count, gather-index,
shape positivity or padding-capacity assumptions are added here. -/
def domain (c : Config) (xs : Values α c) : Prop :=
  ∀ batch, batch<c.Z → length c xs batch≤c.T

#specNonCircular reads avoiding [FP.Structural.exec,runSchedule,Whole.runPrograms,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelIO

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole
noncomputable section

/-- All eight original arrays as named data. Arbitrary Nat-domain suffixes are
retained, so unchecked sparse gathers and aliasing keep their original scope. -/
structure LogicalInputs where
  queryArray : Nat → ℝ
  keyArray : Nat → ℝ
  valueArray : Nat → ℝ
  sequenceLength : Nat → Nat
  blockCounts : Nat → Nat
  blockOffsets : Nat → Nat
  columnCounts : Nat → Nat
  columnIndices : Nat → Nat

/-- Naming arrays is a bijection of complete inputs, without a new bound. -/
def namedInputs (c : Config) : Values ℝ c ≃ LogicalInputs where
  toFun := fun xs => ⟨xs (0 : Fin 8), xs (1 : Fin 8), xs (2 : Fin 8),
    xs (3 : Fin 8), xs (4 : Fin 8), xs (5 : Fin 8), xs (6 : Fin 8), xs (7 : Fin 8)⟩
  invFun := fun xs => Fin.cases xs.queryArray
    (Fin.cases xs.keyArray (Fin.cases xs.valueArray
      (Fin.cases xs.sequenceLength (Fin.cases xs.blockCounts
        (Fin.cases xs.blockOffsets (Fin.cases xs.columnCounts
          (Fin.cases xs.columnIndices (fun port => Fin.elim0 port))))))))
  left_inv := by intro xs; funext port off; fin_cases port <;> rfl
  right_inv := by intro xs; cases xs; rfl

/-- Tensor coordinates are connected to the original public gather arrays here. -/
def LogicalInputs.query (xs : LogicalInputs) (c : Config) (hz token channel : Nat) : ℝ :=
  xs.queryArray (logicalAddress c hz token channel)
def LogicalInputs.key (xs : LogicalInputs) (c : Config) (hz token channel : Nat) : ℝ :=
  xs.keyArray (logicalAddress c hz token channel)
def LogicalInputs.value (xs : LogicalInputs) (c : Config) (hz token channel : Nat) : ℝ :=
  xs.valueArray (logicalAddress c hz token channel)
def LogicalInputs.blockCount (xs : LogicalInputs) (c : Config) (hz query : Nat) : Nat :=
  xs.blockCounts (hz*c.NR+query/c.BM)
def LogicalInputs.columnCount (xs : LogicalInputs) (c : Config) (hz query : Nat) : Nat :=
  xs.columnCounts (hz*c.NR+query/c.BM)
def LogicalInputs.blockOffset (xs : LogicalInputs) (c : Config) (hz query block : Nat) : Nat :=
  xs.blockOffsets ((hz*c.NR+query/c.BM)*c.NS+block)
def LogicalInputs.column (xs : LogicalInputs) (c : Config) (hz query slot : Nat) : Nat :=
  xs.columnIndices ((hz*c.NR+query/c.BM)*c.NV+slot)

/-- The complete sole output tensor in batch-head, query and channel coordinates. -/
def logicalOutputs (c : Config) : ChannelValues ℝ (outputs c) ≃
    (Fin (c.Z*c.H) → Fin c.T → Fin c.BD → ℝ) where
  toFun := fun ys hz row col => ys (0 : Fin 1) (hz,row,col,PUnit.unit)
  invFun := fun ys => Fin.cases (fun i => ys i.1 i.2.1 i.2.2.1) (fun port => Fin.elim0 port)
  left_inv := by
    intro ys
    funext port i
    fin_cases port
    rcases i with ⟨hz,row,col,⟨⟩⟩
    rfl
  right_inv := fun _ => rfl

end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelIO

namespace VeriTile.Bench.Repairs.MixedSparseAttention.RealIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole
noncomputable section

def lifted (c : Config) (xs : KernelIO.Values ℝ c) : KernelIO.Values Real.R c :=
  fun i off=>mapValue some (KernelIO.inputs c)[i].dtype (xs i off)

/-- Real-valued public arrays are encoded in the existing extended-Real source
carrier. The source itself retains its original typed loads/casts/stores. -/
def io (c : Config) (work : List Program) : ArrayProgramIO ℝ Real.R :=
  KernelIO.ioWith Real.algebra Real.dotModel c work (lifted c) some (fun x => x.unbotD 0)
    (fun | ⟨0,_⟩,_,i=>KernelIO.address c i)
    (fun | ⟨0,_⟩,xs,i=>KernelIO.active c (lifted c xs) i)

def domain (c : Config) (xs : KernelIO.Values ℝ c) := KernelIO.domain c (lifted c xs)

/-- Independent sparse softmax over the original occurrence list. Duplicate
indices retain multiplicity; sparse columns have no invented upper bound. -/
def mathematical (c : Config) (xs : KernelIO.Values ℝ c) :
    ChannelValues ℝ (KernelIO.outputs c) := fun
  | ⟨0,_⟩,i=>
    let hz:=i.1.val
    let n:=i.2.1.val
    let d:=i.2.2.1.val
    let row:=hz*c.NR+n/c.BM
    Whole001.MathTarget.expected c.BD c.BN (KernelIO.length c xs (hz/c.H)) n
      (KernelIO.blocks c xs row) (KernelIO.counts c xs row) c.scale
      (fun k=>KernelIO.query c xs (logicalAddress c hz n k))
      (fun j k=>KernelIO.key c xs (logicalAddress c hz j k))
      (fun j k=>KernelIO.value c xs (logicalAddress c hz j k))
      (fun b=>KernelIO.offsets c xs (row*c.NS+b))
      (fun j=>KernelIO.columns c xs (row*c.NV+j)) d

#specNonCircular mathematical avoiding [FP.Structural.exec,runSchedule,Whole.runPrograms,
  Source.finalValues,Source.update,State.mem]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.RealIO

namespace VeriTile.Bench.Repairs.MixedSparseAttention.RealIO
open VeriTile Triton FP.Structural
open Whole002 Source Whole
noncomputable section

/-- The original sequence-length domain, expressed on the same complete named arrays. -/
def logicalDomain (c : Config) (xs : KernelIO.LogicalInputs) : Prop :=
  ∀ batch, batch<c.Z → xs.sequenceLength batch≤c.T

/-- Complete original array domain, invocation order, access footprints and frame. -/
def logicalIO (c : Config) (work : List Program) :
    LogicalDomainIO (KernelIO.Values ℝ c) (ChannelValues ℝ (KernelIO.outputs c))
      KernelIO.LogicalInputs (Fin (c.Z*c.H) → Fin c.T → Fin c.BD → ℝ)
      (io c work).ImplementsOn where
  inputs := KernelIO.namedInputs c
  outputs := KernelIO.logicalOutputs c

end
end VeriTile.Bench.Repairs.MixedSparseAttention.RealIO
