import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Machine
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Physical
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Checked
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MixedSparseAttention.Whole002 Source Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
/-- The finite tagged-region boundary of the source execution model. Base
addresses are the allocator/typed-word ABI binding, not a raw-byte interpreter.
The fresh output interval is disjoint from all existing input allocations. -/
structure Contract {α : Type} [Inhabited α] (c : Config) (before : Physical.Store α)
    (base : RegionName → Nat) : Prop where
  computeDtype : c.storage = .fp16
  shape : Shape c
  separated : Separated c
  fresh : before.capacity c.out=0
  metadata : Metadata c before.extend
  storage : Access.Storage c (Physical.allocatedCapacity c before)
  machine : Machine.Domain c base
  freshBytes : ∀ r width,(r,width)∈Machine.bindings c → r≠c.out →
    base r+width*before.capacity r≤base c.out ∨ base c.out+4*tensorSize c≤base r

theorem bm_pos {α : Type} [Inhabited α] {c : Config} {before : Physical.Store α}
    {base : RegionName → Nat} (h : Contract c before base) : 0<c.BM := by
  have hh:=h.shape.bm
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hh
  omega
theorem bn_pos {α : Type} [Inhabited α] {c : Config} {before : Physical.Store α}
    {base : RegionName → Nat} (h : Contract c before base) : 0<c.BN := by
  have hh:=h.shape.bn
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hh
  omega

/-- Runtime descriptor bound to the same output region that the source stores. -/
structure Handle where
  pointer : Nat
  byteLength : Nat
  tensor : Tensor

def handle (c : Config) (base : RegionName → Nat) : Handle :=
  ⟨base c.out,4*tensorSize c,returnedTensor c⟩

def byteAddress (c : Config) (base : RegionName → Nat) (hz n d : Nat) : Nat :=
  (handle c base).pointer+4*logicalAddress c hz n d

theorem output_byte_bound {α : Type} [Inhabited α] (c : Config) (before : Physical.Store α)
    (base : RegionName → Nat) (h : Contract c before base)
    (hz n d : Nat) (hh : hz<c.Z*c.H) (hn : n<c.T) (hd : d<c.BD) :
    byteAddress c base hz n d+3<(handle c base).pointer+(handle c base).byteLength ∧
      byteAddress c base hz n d+3<18446744073709551616 := by
  have hi:=Whole.dense_bound (c.Z*c.H) c.T c.BD hz n d hh hn hd
  have he:=Machine.dense_bound c hz n d (Nat.le_of_lt hh) (by omega) (Nat.le_of_lt hd)
  have hb:=(h.machine.addresses c.out 4 (by simp [Machine.bindings])).2
  unfold byteAddress handle logicalAddress tensorSize at *
  dsimp only at *
  constructor <;> omega


#axiomsClean output_byte_bound
#specNonCircular Contract avoiding [FP.Structural.exec,Whole.invoke,Source.finalValues]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Checked
