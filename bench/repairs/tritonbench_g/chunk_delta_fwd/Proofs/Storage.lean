import bench.repairs.tritonbench_g.chunk_delta_fwd.Contract
import bench.repairs.tritonbench_g.chunk_delta_fwd.Proofs.NativeGraph
import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.ByteMemory
import bench.repairs.tritonbench_g.chunk_gla_simple.Proofs.ScalarGraph

namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked
open VeriTile Triton FP.Structural
noncomputable section
namespace Bytes
open VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory
abbrev Memory := VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory.Memory
abbrev Word := BitVec 32
abbrev read := VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory.read
abbrev stores := VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory.stores

def Outside (t : Tensor) (address : Nat) := address<t.byteBase ∨ t.byteBase+4*t.count≤address

theorem read_stores (memory : Memory) (t : Tensor) (values : Nat→Word)
    (i : Nat) (hi : i<t.count) : read (stores memory t.byteBase t.count values) (t.byteBase+4*i)=values i := by
  dsimp only [stores,read]
  rw [stores_render];exact render_read _ _ _ _ i hi

theorem frame (memory : Memory) (t : Tensor) (values : Nat→Word)
    (address : Nat) (outside : Outside t address) : stores memory t.byteBase t.count values address=memory address := by
  dsimp only [stores]
  rw [stores_render];exact render_frame _ _ _ _ address outside

theorem empty (memory : Memory) (t : Tensor) (values : Nat→Word) (h : t.count=0) :
    stores memory t.byteBase t.count values=memory := by simp [h,stores,VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory.stores]

theorem separate_frame (memory : Memory) (a b : Tensor) (values : Nat→Word)
    (separate : Separate a b) (j : Nat) (hj : j<4*b.count) :
    stores memory a.byteBase a.count values (b.byteBase+j)=memory (b.byteBase+j) := by
  rcases separate with ha|hb|hab|hba
  · rw [empty memory a values ha]
  · omega
  · apply frame;right;omega
  · apply frame;left;omega

theorem separate_read (memory : Memory) (a b : Tensor) (values : Nat→Word)
    (separate : Separate a b) (i : Nat) (hi : i<b.count) :
    read (stores memory a.byteBase a.count values) (b.byteBase+4*i)=read memory (b.byteBase+4*i) := by
  unfold read VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory.read
  congr 1
  funext j
  rw [Nat.add_assoc]
  apply separate_frame memory a b values separate
  have:=j.isLt
  omega

theorem Separate.symm {a b : Tensor} (h : Separate a b) : Separate b a := by
  unfold Checked.Separate at *;tauto
end Bytes

/-- The successful ordinary allocator supplies only tensor metadata and
freshness. Pointer room is checked after allocation, not assumed here. -/
structure Allocations (r : Request) where
  (history corrected : Tensor)
  historyValid : history.valid [r.B,r.H,r.dims.NT*r.K,r.V] r.key.device
  correctedValid : corrected.valid [r.B,r.H,r.T,r.V] r.key.device
  fresh : ∀ t∈[history,corrected],∀ input∈r.all,Separate t input ∧ t.identity≠input.identity
  pair : Separate history corrected
  identity : history.identity≠corrected.identity

def Allocations.check (r : Request) (a : Allocations r) : Bool :=
  decide (Pointer a.history (formedExtent r.dims) ∧ Pointer a.corrected (formedExtent r.dims))

theorem Allocations.history_count (r : Request) (a : Allocations r) :
    a.history.count=r.B*r.H*r.dims.NT*r.K*r.V := by
  simp only [Tensor.count,a.historyValid.1,List.prod_cons,List.prod_nil]
  ring

theorem Allocations.corrected_count (r : Request) (a : Allocations r) :
    a.corrected.count=r.B*r.H*r.T*r.V := by
  simp only [Tensor.count,a.correctedValid.1,List.prod_cons,List.prod_nil]
  ring

theorem Request.state_count (r : Request) (checked : r.check=Bool.true)
    (t : Tensor) (ht : t∈r.initial.toList++r.final.toList) : t.count=r.B*r.H*r.K*r.V := by
  have h:=(r.check_iff.mp checked).2.2.2.2.2.2.2.2.2.2.2.1 t ht
  simp only [Tensor.count,h.1,List.prod_cons,List.prod_nil]
  ring

def descriptor (r : Request) (a : Allocations r) (region : RegionName) : Option Tensor :=
  if region="key" then some r.key else if region="weight" then some r.weight else
  if region="value" then some r.value else if region="initial" then r.initial else
  if region="history" then some a.history else if region="corrected" then some a.corrected else
  if region="final" then r.final else none

def bits (r : Request) (a : Allocations r) (before : Bytes.Memory) : RegionName→Nat→BitVec 32 :=
  fun region i=>match descriptor r a region with
    | some t=>Bytes.read before (t.byteBase+4*i)
    | none=>0

def input (r : Request) (a : Allocations r) (before : Bytes.Memory) := Native.inputs (bits r a before)

theorem input_word (r : Request) (a : Allocations r) (before : Bytes.Memory)
    (region : RegionName) (t : Tensor) (bound : descriptor r a region=some t) (i : Nat) :
    (input r a before region i).read .real = .bits (Bytes.read before (t.byteBase+4*i)) := by
  simp [input,Native.inputs,bits,bound,Cell.read_same]

abbrev Primitives := VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph.WordPrimitives

def source (r : Request) (a : Allocations r) (before : Bytes.Memory) :=
  gridMemory Native.algebra Native.dot r.params (grid r.params) (input r a before) (input r a before)

def words (I : Primitives) (memory : Memory VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.Expr)
    (region : RegionName) (i : Nat) : BitVec 32 := (Native.lower ((memory region i).read .real)).eval I.isa

def publish (I : Primitives) (r : Request) (a : Allocations r) (before : Bytes.Memory)
    (memory : Memory VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.Expr) : Bytes.Memory :=
  let history:=Bytes.stores before a.history.byteBase a.history.count (words I memory "history")
  let corrected:=Bytes.stores history a.corrected.byteBase a.corrected.count (words I memory "corrected")
  match r.final with
  | none=>corrected
  | some t=>Bytes.stores corrected t.byteBase t.count (words I memory "final")

def returnedBytes (I : Primitives) (r : Request) (a : Allocations r) (before : Bytes.Memory) :=
  publish I r a before (source r a before)

#axiomsClean Bytes.separate_read
#axiomsClean input_word
#specNonCircular Allocations avoiding [source,runGrid,gridMemory,states]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Checked
