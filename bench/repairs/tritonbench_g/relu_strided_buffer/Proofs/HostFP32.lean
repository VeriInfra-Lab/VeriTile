import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Host
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeFP32

namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer
open HostPlan SignedView SignedGrid Whole
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

/-- Every FP32 word: strict ordered positive selection or +0. The checked
signed-view domain entails all output values, cells/bytes frame, full native
index/mask/access bounds and physical-pointer binding. No numerical laws. -/
specification relu_strided_buffer_checked_fp32 (rest : Algebra (FP.Value .fp32)) (x y : ViewMetadata)
    (admitted : checkInputs x y = .ok ()) (api : AllocationAPI x y)
    (s : State (FP.Value .fp32)) (xs : Nat → FP.Value .fp32)
    (values : ∀ i, i < x.n → s.mem x.buffer.region ((x.buffer.view x.stride).address i) = .mk .fp32 (xs i)) :
    let h := (checkInputs_iff _ _).mp admitted
    let buffers := nativeBuffers x y h api
    ∃ final, runChecked (fp32Backend rest) x y Bool.true s = .ok (.returned y.buffer final) ∧
      (∀ i, i < x.n → final.mem y.buffer.region ((y.buffer.view y.stride).address i) =
        Cell.mk .fp32 (fp32Result xs i)) ∧
      (∀ i, i < x.n → final.mem x.buffer.region ((x.buffer.view x.stride).address i) =
        s.mem x.buffer.region ((x.buffer.view x.stride).address i)) ∧
      (∀ r a, (r ≠ y.buffer.region ∨ ∀ i, i < x.n → a ≠ (y.buffer.view y.stride).address i) →
        final.mem r a = s.mem r a) ∧
      (∀ i, i < x.n → (x.buffer.view x.stride).address i < x.capacity ∧
        (y.buffer.view y.stride).address i < y.capacity) ∧
      NativeAccess x.buffer y.buffer x.n x.stride y.stride buffers ∧
      SourceAddressLowering x.buffer y.buffer x.n x.stride y.stride buffers ∧
      (SignedByteAddress.dataPtr64 x.allocationBase (x.buffer.view x.stride)).toNat = x.pointer.toNat ∧
      (SignedByteAddress.dataPtr64 y.allocationBase (y.buffer.view y.stride)).toNat = y.pointer.toNat ∧
      (∀ i, i < x.n → ∀ byte, byte < 4 →
        memoryByte final y.buffer.region ((y.buffer.view y.stride).address i) byte = wordByte (fp32Result xs i) byte) ∧
      (∀ r a, (r ≠ y.buffer.region ∨ ∀ i, i < x.n → a ≠ (y.buffer.view y.stride).address i) →
        ∀ byte, byte < 4 → memoryByte final r a byte = memoryByte s r a byte) ∧
      (∀ i j u v, i < x.n → j < x.n → u < 4 → v < 4 →
        x.allocationBase+4*(x.buffer.view x.stride).address i+u ≠
          y.allocationBase+4*(y.buffer.view y.stride).address j+v) := by
  dsimp only
  let h := (checkInputs_iff _ _).mp admitted
  let buffers := nativeBuffers x y h api
  have outputShape : y.buffer.shape = [x.n] := h.output.rank.1.trans (congrArg List.singleton h.shape.symm)
  have inputValid := ViewContract.valid x h.input
  have outputValid : (y.buffer.view y.stride).Valid x.n y.capacity := h.shape.symm ▸ ViewContract.valid y h.output
  have separated := input_output_footprints x y h api
  obtain ⟨final,ran,result,inputFrame,frame,bounds⟩ := fp32_complete rest x.buffer y.buffer
    h.input.extent.1 x.stride y.stride h.input.rank.1 outputShape h.input.rank.2 h.output.rank.2
    inputValid outputValid h.outputStride separated s xs values
  refine ⟨final,(checked_eq_source _ x y admitted _ s).trans (congrArg Except.ok ran),
    result,inputFrame,frame,bounds,
    native_access x.buffer y.buffer h.input.extent.1 h.input.extent.2 x.stride y.stride inputValid outputValid buffers,
    source_address_lowering x.buffer y.buffer h.input.extent.1 h.input.extent.2 x.stride y.stride inputValid outputValid buffers,
    actual_pointer x h.input api.inputAllocation,actual_pointer y h.output api.outputAllocation,?_,?_,?_⟩
  · intro i hi byte _; exact output_bytes final _ _ (fp32Result xs i) (result i hi) byte
  · intro r a outside byte _; exact frame_bytes s final r a (frame r a outside) byte
  · intro i j u v hi hj hu hv
    exact input_output_bytes_disjoint x.buffer y.buffer x.stride y.stride inputValid outputValid
      separated buffers hi hj hu hv
#axiomsClean relu_strided_buffer_checked_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluStridedBuffer
