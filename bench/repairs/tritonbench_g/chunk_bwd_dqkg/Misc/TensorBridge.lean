import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelResults
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelFrames

/-! Optional Tensor readback, input-word binding and backing-byte frame
corollaries. Complete Tensor applications retain the common source-execution foundation used by the public IO proofs. -/
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.TensorBridge
open VeriTile.Triton FP.Structural Whole KernelTensor KernelContract
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section

theorem kernel_real_tensor {Store Byte : Type} (api : Memory.API ℝ Store Byte) (b : Buffers c)
    (store : Store) (final : BlockState) (r : RegionName) (i : Nat) (active : Writes c r i) :
    api.read (Shared.commit api (placement b) KernelReal.observe (KernelReal.image final) store)
      ((b.tensor r).storage.address ((b.tensor r).physicalIndex i)) = final.readMem r i :=
  KernelReal.tensor_value api b store final r i active

theorem kernel_native_inputs (isa : KernelNativeCarrier.Instructions)
    (route : KernelNativeCarrier.Routing) (b : Buffers c) (read : Nat → KernelNativeCarrier.Word)
    (r : RegionName) (i : Nat) (hi : i < (b.tensor r).metadata.numel) :
    KernelNativeCarrier.observe isa route ((KernelNative.initial b read).mem r i) =
      (VeriTile.Torch.FP32.logicalValue (fun a => ⟨read a⟩) (b.tensor r) (b.laws r) i).bits :=
  KernelNative.input_words isa route b read r i hi

theorem kernel_input_frame {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (b : Buffers c) (observe : RegionName → Nat → Cell α → β) (final : State α) (store : Store)
    (r : Role) (hr : r ∈ KernelFrames.inputRoles) (i : Nat)
    (hi : i < (b.tensor (region r)).metadata.numel) (byte : Fin 4) :
    api.byte (Shared.commit api (placement b) observe final store) ((b.view (region r)).address i+byte.val) =
      api.byte store ((b.view (region r)).address i+byte.val) :=
  KernelFrames.input_bytes api b observe final store r hr i hi byte


/-- The original complete Real Tensor application, with its full storage image. -/
theorem kernel_real {Store Byte : Type} (api : Memory.API ℝ Store Byte)
    (b : Buffers c) (store : Store) (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    ∃ final, Whole.runGrid c order (KernelReal.initial b (api.read store)) = some final ∧
      KernelReal.Result c (KernelReal.initial b (api.read store)) final ∧
      Shared.TensorImage api (placement b) KernelReal.observe (KernelReal.image final) store := by
  obtain ⟨final,he,hr⟩ := KernelResults.kernel_real c
    (KernelReal.initial b (api.read store)) order ho
  exact ⟨final,he,hr,Shared.complete api (placement b) KernelReal.observe (KernelReal.image final) store⟩

open KernelNativeCarrier KernelNative in
/-- The original complete native Tensor application, including compiler and scalar ABI bindings. -/
theorem kernel_native {Store Byte : Type} (api : Memory.API Word Store Byte)
    (platform : KernelCompiler.Platform) (b : Buffers c) (binding : KernelCompiler.Binding platform b)
    (conversion : KernelCompiler.ScaleConversion) (argument : KernelCompiler.ScaleArgument)
    (scale : KernelCompiler.LaunchScale conversion argument c) (store : Store)
    (order : List (CTA c)) (ho : order.Perm (allCTAs c)) :
    let isa := platform.instructions binding.object
    let route := binding.decoded.routing
    ∃ final, runQuoted binding.decoded.source isa route scale.value.word.bits c order (initial b (api.read store)) = some final ∧
      Result isa route scale.value.word.bits c (initial b (api.read store)) final ∧
      Shared.TensorImage api (placement b) (fun _ _ => observe isa route) final store ∧
      (∀ r i, Writes c r i →
        (VeriTile.Torch.FP32.logicalValue (fun addr => ⟨api.read (Shared.commit api (placement b)
          (fun _ _ => observe isa route) final store) addr⟩) (b.tensor r) (b.laws r) i).bits = observe isa route (final.mem r i)) := by
  dsimp only
  obtain ⟨final,he,hr⟩ := KernelResults.kernel_native
    (platform.instructions binding.object) binding.decoded.routing scale.value.word.bits c
    (initial b (api.read store)) order ho
  refine ⟨final,?_,hr,Shared.complete api (placement b) _ final store,?_⟩
  · simpa only [binding.quote] using he
  · exact Shared.native_logical_output api (placement b) _ final store

#axiomsClean kernel_real
#axiomsClean kernel_native

#axiomsClean kernel_real_tensor
#axiomsClean kernel_native_inputs
#axiomsClean kernel_input_frame
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.TensorBridge
