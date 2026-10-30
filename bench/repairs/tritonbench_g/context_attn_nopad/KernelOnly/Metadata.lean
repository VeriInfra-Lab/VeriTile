import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Safety

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section

theorem index_pointer (x : IndexTensor) (laws : IndexLaws x)
    (hc : x.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<x.metadata.numel) :
    x.storageBase+4*x.physicalIndex i=x.metadata.dataPtr.toNat+4*i := by
  have capacity := laws.inBounds i hi
  rw [laws.contiguous hc i hi] at capacity
  have span := laws.span
  have bound : x.storageBase+4*x.storageOffset<2^64 := by omega
  rw [laws.contiguous hc i hi,laws.pointer (by omega)]
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound]
  omega

theorem logical_metadata (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (b : Nat) (hb : b<r.call.batches) :
    (load32 heap (r.start.storageBase+4*r.start.physicalIndex b)).toInt=(r.call.start b : Int) ∧
    (load32 heap (r.length.storageBase+4*r.length.physicalIndex b)).toInt=(r.call.length b : Int) := by
  have start := h.checks.1.2.1 (views r).start (by simp)
  have length := h.checks.1.2.1 (views r).length (by simp)
  have sb : b<r.start.metadata.numel := by
    simpa [Torch.TensorMetadata.numel,show r.start.metadata.shape=[r.call.batches] from start.1] using hb
  have lb : b<r.length.metadata.numel := by
    simpa [Torch.TensorMetadata.numel,show r.length.metadata.shape=[r.call.batches] from length.1] using hb
  rw [index_pointer r.start h.startAPI start.2.2 b sb,index_pointer r.length h.lengthAPI length.2.2 b lb]
  exact h.checks.2.2.2.1 ⟨b,hb⟩

theorem metadata_byte_offset (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (b : Nat) (hb : b<r.call.batches) :
    (Safety.word b*Safety.word 4).toInt=((b*4 : Nat) : Int) := by
  rw [Safety.word,Safety.word,←BitVec.ofNat_mul]
  apply Safety.word_exact
  have count := h.checks.2.2.1.2.2.1
  omega

#axiomsClean logical_metadata
#axiomsClean metadata_byte_offset
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
