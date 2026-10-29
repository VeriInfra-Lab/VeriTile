import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelWhole
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelAddresses

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelStores
open VeriTile.Triton FP.Structural KernelInputs KernelWhole
open VeriTile.Bench.TritonBenchG.QuantizeKvTransform
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section

/-- The int8 source store observes precisely the selected conversion's low
byte, including NaNs, infinities and values outside the portable i8 domain. -/
theorem signed_byte_observation (x : Word) :
    BitVec.ofInt 8 (PtxSigned.signedByte x) = PtxSigned.byte x := by
  simp [PtxSigned.signedByte]

/-- Admission and the existing physical metadata binding make the typed
destination read exactly the signed INT64 interpretation of its eight bytes. -/
theorem signed_metadata (m : Inputs) (words : Nat → MetadataWord)
    (h : InputContract m words) (heap : Heap) (binding : TensorBinding m words heap)
    (initial : State Word) (reads : NativeReads m words heap initial)
    (token : Nat) (ht : token < (request m words).tokens) :
    (initial.mem "destination" token).read .int =
      (Native.signed (loadDest heap (base m .destination + 8 * token))).toInt := by
  rw [reads.destination token ht, binding.metadata token ht]
  exact (Native.signed_exact _ ((checked_domain m words h).destination_int64 token ht)).symm

/-- Each final typed store is observed at its caller Tensor coordinate and
exact native INT64 byte address, with its actual int8 or two-byte FP16 codec.
This is a store observation, not an additional physical-heap interpreter.
The full typed exterior frame remains part of TensorResult. -/
def NativeStores (a : KernelNativeProfile.Arithmetic) (m : Inputs)
    (words : Nat → MetadataWord) (heap : Heap) (final : State Word) : Prop :=
  (∀ token head column, token < (request m words).tokens →
    head < (request m words).heads → column < (request m words).columns →
    let slot := (request m words).destination token
    let offset := KernelResult.codeOffset (parameters m words) slot head column
    (Native.pointer (base m .output)
      (Native.outputIndex (request m words) slot head column) 1).toNat =
        base m .output + offset ∧
    BitVec.ofInt 8 ((final.mem "output" offset).read .int) =
      BitVec.ofInt 8 (nativeCode a m words heap token head column)) ∧
  (∀ token head, token < (request m words).tokens → head < (request m words).heads →
    let slot := (request m words).destination token
    let offset := KernelResult.scaleOffset (parameters m words) slot head
    ∀ byte : Fin 2,
      (Native.pointer (base m .scale)
        (Native.scaleIndex (request m words) slot head) 2).toNat + byte.val =
          base m .scale + 2 * offset + byte.val ∧
      halfByte (MixedBits.half ((final.mem "scale" offset).read .fp16)) byte =
        halfByte (MixedBits.half (nativeScale a m words heap token head)) byte)

theorem of_result (a : KernelNativeProfile.Arithmetic) (m : Inputs)
    (words : Nat → MetadataWord) (h : InputContract m words)
    (heap : Heap) (binding : TensorBinding m words heap) (initial final : State Word)
    (result : TensorResult m words (nativeCode a m words heap)
      (nativeScale a m words heap) initial final) :
    NativeStores a m words heap final := by
  refine ⟨?_, ?_⟩
  · intro token head column ht hh hc
    refine ⟨(KernelAddresses.pointers m words h (fun _ => 0) heap binding
      token head column ht hh hc).2.1, ?_⟩
    dsimp only
    rw [result.1 token head column ht hh hc]
    rfl
  · intro token head ht hh
    dsimp only
    intro byte
    refine ⟨(KernelAddresses.pointers m words h (fun _ => 0) heap binding
      token head 0 ht hh (width_positive m words h)).2.2 byte, ?_⟩
    dsimp only
    rw [result.2.1 token head ht hh]
    rfl

/-- Both scale bytes recover the actual FP16 rounding result. The FP32 carrier
used by typed execution does not insert another rounding or change any bit. -/
theorem scale_byte_observation (a : KernelNativeProfile.Arithmetic) (m : Inputs)
    (words : Nat → MetadataWord) (heap : Heap) (token head : Nat)
    (hh : head < (parameters m words).height) (byte : Fin 2) :
    halfByte (MixedBits.half (nativeScale a m words heap token head)) byte =
      halfByte (KernelNativeResult.scale a (parameters m words)
        (tensorWords m words heap token) ⟨head, hh⟩) byte := by
  simp [nativeScale, hh, MixedBits.half_embedding]

#axiomsClean signed_byte_observation
#axiomsClean signed_metadata
#axiomsClean of_result
#axiomsClean scale_byte_observation
#specNonCircular NativeStores avoiding [KernelSource.kernel, FP.Structural.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelStores
