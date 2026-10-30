import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Native
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Safety

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Program
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

theorem index_contiguous (r : Request) (heap : ByteMemory) (h : Contract r heap) (role : IndexRole) :
    (role.tensor r).metadata.isContiguous=Bool.true := by
  have vector := h.checks.1.2.2.1
  have pages := h.checks.1.2.2.2.1.2.2.1
  cases role with
  | start => exact (vector (views r).start (by simp)).2.2
  | sequence => exact (vector (views r).sequence (by simp)).2.2
  | prompt => exact (vector (views r).prompt (by simp)).2.2
  | request => exact (vector (views r).request (by simp)).2.2
  | pages => exact pages

/-- Every vector and page-table logical INT32 word is read from its actual
Tensor physical index, including nonzero storage offsets and shared backing. -/
theorem logical_metadata (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (role : IndexRole) (i : Nat) (hi : i<(role.tensor r).metadata.numel) :
    (load32 heap ((role.tensor r).storageBase+4*(role.tensor r).physicalIndex i)).toInt=
      indexValues r heap role i := by
  rw [index_pointer _ (h.indexAPI role) (index_contiguous r heap h role) i hi]
  rfl

/-- Typed source metadata operands and the native widened load consume the
same original physical word. There is no separately assumed metadata copy. -/
theorem source_metadata (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (role : IndexRole) (i : Nat) (hi : i<(role.tensor r).metadata.numel) :
    ((initial r heap).mem role.region i).read .int=
      (load32 heap ((role.tensor r).storageBase+4*(role.tensor r).physicalIndex i)).toInt ∧
    (load32 heap ((role.tensor r).storageBase+4*(role.tensor r).physicalIndex i)).signExtend 64=
      Integer.sword (indexValues r heap role i) := by
  rw [logical_metadata r heap h role i hi]
  constructor
  · cases role <;> rfl
  · change Integer.sword ((load32 heap ((role.tensor r).storageBase+4*(role.tensor r).physicalIndex i)).toInt)=_
    rw [logical_metadata r heap h role i hi]

/-- The native addptr residue is the actual physical INT32 Tensor address. -/
theorem index_native_pointer (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (role : IndexRole) (i : Nat) (hi : i<(role.tensor r).metadata.numel) :
    (Integer.ptr (role.tensor r).metadata.dataPtr.toNat 4 (Integer.word i)).toNat=
      (role.tensor r).storageBase+4*(role.tensor r).physicalIndex i := by
  have pointer := index_pointer _ (h.indexAPI role) (index_contiguous r heap h role) i hi
  have bound := (h.indexAPI role).inBounds i hi
  have span := (h.indexAPI role).span
  rw [Integer.ptr_modular,Integer.unsigned _ (by omega)]
  exact pointer.symm

/-- Unconditional metadata vector byte offsets are signed64 exact. -/
theorem metadata_byte_offset (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (idx : GridIndex r.call.grid) :
    (Integer.word 4*Integer.batchBits r.call.layout (r.call.launchAt idx)).toInt=
      (4*Integer.batchAt r.call idx : Nat) := by
  have finite := Safety.machine r heap h
  rw [(Integer.launch_exact r.call.layout (r.call.launchAt idx)
    (Integer.batchHead_abi r.call Safety.widths finite idx) finite.heads finite.group).1]
  simp only [Integer.word,BitVec.ofNat_mul_ofNat]
  apply Integer.signed
  have batch := Host.batch_bound r.call idx
  have count := h.checks.2.2.1.2.2.1
  omega

#axiomsClean index_native_pointer
#axiomsClean metadata_byte_offset
#axiomsClean logical_metadata
#axiomsClean source_metadata
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly
