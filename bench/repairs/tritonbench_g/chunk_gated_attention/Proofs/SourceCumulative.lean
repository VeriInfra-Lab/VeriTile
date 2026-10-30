import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.SourceAddress

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.SourceCumulative
open VeriTile.Triton HostFP Address SourceAddress
noncomputable section
set_option maxHeartbeats 2000000

structure Coord (c : HostFP.Cumulative) where
  head : Fin c.heads
  chunk : Fin (c.PT/c.BT)
  tile : Fin (c.PS/c.BS)
  time : Fin c.BT
  feature : Fin c.BS

def nativeWord (c : HostFP.Cumulative) (i : Coord c) : BitVec 64 :=
  word i.head.val*word (c.PT*c.PS)+word i.chunk.val*word c.BT*word c.PS+
    word i.tile.val*word c.BS*word 1+word i.time.val*word c.PS+word i.feature.val*word 1

def offset (c : HostFP.Cumulative) (i : Coord c) : Nat :=
  i.head.val*(c.PT*c.PS)+(i.chunk.val*c.BT+i.time.val)*c.PS+i.tile.val*c.BS+i.feature.val

theorem word_eq (c : HostFP.Cumulative) (i : Coord c) : nativeWord c i=word (offset c i) := by
  simp only [nativeWord,offset,word,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  congr 1
  ring

theorem allocated {α : Type}  (x : HostCumulative.Request α) (hx : x.Valid)
    (input output : RegionName) (i : Coord (x.config input output)) :
    offset (x.config input output) i < x.B*x.H*(padded x.T x.BT)*(padded x.S 64) := by
  have ht := SourceAddress.padded_tile (n := x.T) (b := x.BT) (by have := hx.bt;omega) i.chunk.isLt i.time.isLt
  have hf : i.tile.val*x.BS+i.feature.val<padded x.S 64 := by
    have hi := i.tile.isLt
    have hl := i.feature.isLt
    change i.tile.val<padded x.S 64/x.BS at hi
    change i.feature.val<x.BS at hl
    have he : padded x.S 64/x.BS*x.BS=padded x.S 64 := by
      rcases hx.bs with h|h|h <;> rw [h] <;> unfold padded <;> omega
    have hm := Nat.mul_le_mul_right x.BS (Nat.succ_le_iff.mpr hi)
    simp only [Nat.succ_mul,he] at hm
    omega
  have hb := Address.flatten3_bounded i.head.isLt ht hf
  simpa only [offset,Nat.add_assoc,HostCumulative.Request.config,Cumulative.PT,Cumulative.PS] using hb

def inputAccess {α : Type}  (x : HostCumulative.Request α)
    (hx : x.Valid) (input output : RegionName) (p : HostCumulative.Allocated x input output)
    (i : Coord (x.config input output)) : Access p.inputBuffer :=
  ⟨offset _ i,nativeWord _ i,word_eq _ i,lt_of_lt_of_le (allocated x hx input output i) p.inputExtent⟩

def outputAccess {α : Type}  (x : HostCumulative.Request α)
    (hx : x.Valid) (input output : RegionName) (p : HostCumulative.Allocated x input output)
    (i : Coord (x.config input output)) : Access p.outputBuffer :=
  ⟨offset _ i,nativeWord _ i,word_eq _ i,lt_of_lt_of_le (allocated x hx input output i) p.outputExtent⟩

/-- Both source local block pointers have the same element offset, but their
native byte offsets use independently bound input dtype and FP32 output dtype. -/
theorem both_cells {α β : Type}  (x : HostCumulative.Request α)
    (hx : x.Valid) (input output : RegionName) (p : HostCumulative.Allocated x input output)
    (i : Coord (x.config input output)) (mi : Allocation.Cells β p.inputBuffer)
    (mo : Allocation.Cells β p.outputBuffer) :
    mi.physical (p.inputBuffer.nativeAddress (nativeWord _ i))=mi.typed (offset _ i) ∧
    mo.physical (p.outputBuffer.nativeAddress (nativeWord _ i))=mo.typed (offset _ i) :=
  ⟨(inputAccess x hx input output p i).read mi,(outputAccess x hx input output p i).read mo⟩

#print axioms allocated
#print axioms both_cells
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.SourceCumulative
