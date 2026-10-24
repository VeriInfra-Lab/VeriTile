import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelWhole
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelAddresses

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelStores
open VeriTile.Triton FP.Structural KernelInputs KernelWhole
open KernelNativeProfile
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section

/-- The int8 source store observes precisely the selected conversion's low
byte, including NaNs, infinities and values outside the portable i8 domain. -/
theorem signed_byte_observation (x : Word) :
    BitVec.ofInt 8 (PtxSigned.signedByte x) = PtxSigned.byte x := by
  simp [VeriTile.Bench.Repairs.QuantizeCopyKv.PtxSigned.signedByte]

/-- Admission and the existing physical metadata binding make the typed
destination read exactly the signed INT32 promotion of its four bytes. -/
theorem signed_metadata (m : Inputs) (words : Nat → MetadataWord)
    (h : InputContract m words) (heap : Heap) (binding : TensorBinding m words heap)
    (initial : State Word) (reads : NativeReads m words heap initial)
    (token : Nat) (ht : token < (request m words).tokens) :
    (initial.mem "destination" token).read .int =
      (Native.signed (loadDest heap (base m .destination + 4 * token))).toInt := by
  rw [reads.destination token ht, binding.metadata token ht]
  exact (Native.signed_exact _ (lt_trans ((checked_domain m words h).destination_int32 token ht) (by norm_num))).symm

/-- Each final typed store is observed at its caller Tensor coordinate and
exact native INT64 byte address, with its actual int8 or four-byte FP32 codec.
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
    ∀ byte : Fin 4,
      (Native.pointer (base m .scale)
        (Native.scaleIndex (request m words) slot head) 4).toNat + byte.val =
          base m .scale + 4 * offset + byte.val ∧
      scaleByte ((final.mem "scale" offset).read .real) byte =
        scaleByte (nativeScale a m words heap token head) byte)

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
      token head 0 ht hh h.width.2).2.2 byte, ?_⟩
    dsimp only
    rw [result.2.1 token head ht hh]
    rfl

/-- Construct the same complete native source result and observe every store
at its checked pointer, retaining the original full typed frame. -/
theorem native (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (m : Inputs) (words : Nat → MetadataWord) (h : InputContract m words)
    (schedule : Schedule (request m words)) (heap : Heap) (binding : TensorBinding m words heap) (initial : State Word)
    (reads : NativeReads m words heap initial) :
    ∃ final,
      KernelGrid.run (KernelNativeProfile.backend a rest) (parameters m words)
        schedule.order initial=some final ∧
      TensorResult m words (nativeCode a m words heap) (nativeScale a m words heap) initial final ∧
      NativeStores a m words heap final := by
  obtain ⟨final,run,result⟩ := KernelWhole.native a rest m words h schedule heap binding initial reads
  exact ⟨final,run,result,of_result a m words h heap binding initial final result⟩

#axiomsClean signed_byte_observation
#axiomsClean signed_metadata
#axiomsClean of_result
#axiomsClean native
#specNonCircular NativeStores avoiding [KernelSource.kernel, FP.Structural.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelStores
