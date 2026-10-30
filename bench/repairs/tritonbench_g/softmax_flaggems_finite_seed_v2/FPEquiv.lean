import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Proofs.Binding
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Proofs.Replica32
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.FPEquiv

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open Native
noncomputable section
abbrev Allocated := SoftmaxFlaggemsFiniteSeed.FPEquiv.Allocated

/-- Complete forward quotation, all branches and original launch geometry,
consumer-indexed scalar native graph, allocated Tensor words, bytes and frame.
All raw words remain admitted; no numerical equality between replicas is used. -/
theorem checked_forward_graph (x y : VeriTile.Torch.TensorMetadata) (dim : Int)
    (dtype : Option VeriTile.Torch.DType) (admitted : ForwardContract x dim dtype) (allocated : Allocated x y)
    (ax : Fin x.shape.length) (hnorm : normalizeDim x.shape dim=some ax)
    (sms : Nat) (waves : WavesAboveOne) (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (memory : RegionName → Nat → Word) (before : Binding.Bytes) :
    Spec.Real (checkForward x dim dtype=.ok () ∧ normalizeDim x.shape dim=some ax ∧
      ∃ result, Tensor.forward ops scalar R S x y ax "x" "out" (forwardConfig waves (dimensions x.shape ax) sms) memory=some result ∧
        result.tensor=y ∧ result.tensor.shape=x.shape ∧ result.tensor.device=x.device ∧ result.tensor.dtype=.floating .fp32 ∧
        let expected := Tensor.forwardWords ops scalar R S "x" "out" (dimensions x.shape ax) (forwardConfig waves (dimensions x.shape ax) sms) memory
        Tensor.Output "out" x.numel expected memory result ∧
        (∀ i b, i<x.numel → b<4 → Binding.callerBytes "out" y before result.words (y.dataPtr.toNat+4*i+b)=Binding.byteOf (expected i) b) ∧
        (∀ a, a<y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤a → Binding.callerBytes "out" y before result.words a=before a)) := by
  obtain ⟨result,he,hm,output⟩ := Tensor.forward_complete ops scalar R S x y ax "x" "out"
    (forwardConfig waves (dimensions x.shape ax) sms) (forwardConfig_positive waves _ sms).1
    (forwardConfig_positive waves _ sms).2 (forwardConfig_one waves _ sms) memory
  have count : y.numel=x.numel := congrArg List.prod allocated.1
  obtain ⟨bytes,frame⟩ := Binding.output_bytes "out" y before x.numel count _ memory result output
  exact ⟨(SoftmaxFlaggemsFiniteSeed.checkForward_iff x dim dtype).mpr admitted,hnorm,result,he,hm,
    by simpa only [hm] using allocated.1,by simpa only [hm] using allocated.2.1,
    by simpa only [hm] using allocated.2.2.1,output,bytes,frame⟩

/-- Every backward tuning configuration reads actual stored FP32 P/G words,
executes the full quotation, and binds each store's own native replica to the
returned Tensor bytes. This entry does not carry symbolic forward expressions. -/
theorem checked_backward_graph (x grad y : VeriTile.Torch.TensorMetadata) (dim : Int)
    (dtype : Option VeriTile.Torch.DType) (admitted : ForwardContract x dim dtype)
    (gradient : GradientContract x grad) (allocated : Allocated x y)
    (ax : Fin x.shape.length) (hnorm : normalizeDim x.shape dim=some ax) (choice : Fin 6)
    (ops : NativeProfile.Operations) (scalar : ℝ → Word) (R : Reductions) (S : Schedule)
    (memory : RegionName → Nat → Word) (before : Binding.Bytes) :
    Spec.Real (checkForward x dim dtype=.ok () ∧ checkGradient x grad=.ok () ∧ normalizeDim x.shape dim=some ax ∧
      ∃ result, Tensor.backward ops scalar R S x y ax "out" "grad" "in_grad" (backwardConfig (dimensions x.shape ax) choice) memory=some result ∧
        result.tensor=y ∧ result.tensor.shape=x.shape ∧ result.tensor.device=x.device ∧ result.tensor.dtype=.floating .fp32 ∧
        let expected := Tensor.backwardWords ops scalar R S "out" "grad" "in_grad" (dimensions x.shape ax) (backwardConfig (dimensions x.shape ax) choice) memory
        Tensor.Output "in_grad" x.numel expected memory result ∧
        (∀ i b, i<x.numel → b<4 → Binding.callerBytes "in_grad" y before result.words (y.dataPtr.toNat+4*i+b)=Binding.byteOf (expected i) b) ∧
        (∀ a, a<y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤a → Binding.callerBytes "in_grad" y before result.words a=before a)) := by
  obtain ⟨result,he,hm,output⟩ := Tensor.backward_complete ops scalar R S x y ax "out" "grad" "in_grad"
    (backwardConfig (dimensions x.shape ax) choice) (backwardConfig_positive _ choice).1
    (backwardConfig_positive _ choice).2 (backwardConfig_one _ choice) memory
  have count : y.numel=x.numel := congrArg List.prod allocated.1
  obtain ⟨bytes,frame⟩ := Binding.output_bytes "in_grad" y before x.numel count _ memory result output
  exact ⟨(SoftmaxFlaggemsFiniteSeed.checkForward_iff x dim dtype).mpr admitted,
    (SoftmaxFlaggemsFiniteSeed.checkGradient_iff x grad).mpr gradient,hnorm,result,he,hm,
    by simpa only [hm] using allocated.1,by simpa only [hm] using allocated.2.1,
    by simpa only [hm] using allocated.2.2.1,output,bytes,frame⟩

/-- A subsequent kernel can change consumer, reductions and fusion profile;
the previous result is nevertheless loaded as its fixed stored word. -/
theorem tensor_word_handoff (I : Scalar.Instructions) (S : Schedule) (result : Tensor.Result)
    (r : RegionName) (o : Nat) (consumer : Consumer) :
    Scalar.eval I (S consumer) (((entry result.words).mem r o).read .real consumer)=result.words r o := rfl

/-- Public native Tensor endpoint with actual input bytes, one before/post
memory, fresh destination storage, and the entire input backing frame. -/
specification checked_forward (x y : VeriTile.Torch.TensorMetadata) (dim : Int)
    (dtype : Option VeriTile.Torch.DType) (admitted : ForwardContract x dim dtype) (allocated : Allocated x y)
    (ax : Fin x.shape.length) (hnorm : normalizeDim x.shape dim=some ax)
    (sms : Nat) (waves : WavesAboveOne) (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (memory : RegionName → Nat → Word) (before : Binding.Bytes)
    (xStorage yStorage : Binding.Backing) (xWithin : Binding.Within x xStorage) (yWithin : Binding.Within y yStorage)
    (fresh : 0<y.numel → Binding.Separate yStorage xStorage)
    (inputBytes : Binding.InputBytes "x" x memory before) :
    Spec.Real (checkForward x dim dtype=.ok () ∧ normalizeDim x.shape dim=some ax ∧
      Binding.InputLoads R "x" x memory before ∧
      ∃ result, Tensor.forward ops scalar R S x y ax "x" "out" (forwardConfig waves (dimensions x.shape ax) sms) memory=some result ∧
        result.tensor=y ∧ result.tensor.shape=x.shape ∧ result.tensor.device=x.device ∧ result.tensor.dtype=.floating .fp32 ∧
        let expected := Tensor.forwardWords ops scalar R S "x" "out" (dimensions x.shape ax) (forwardConfig waves (dimensions x.shape ax) sms) memory
        let post := Binding.callerBytes "out" y before result.words
        Tensor.Output "out" x.numel expected memory result ∧
        (∀ i b, i<x.numel → b<4 → post (y.dataPtr.toNat+4*i+b)=Binding.byteOf (expected i) b) ∧
        (∀ a, a<y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤a → post a=before a) ∧
        (∀ a, xStorage.base≤a ∧ a<xStorage.base+xStorage.bytes → post a=before a) ∧
        (∀ i, i<x.numel → Binding.wordAt post (x.dataPtr.toNat+4*i)=memory "x" i)) := by
  obtain ⟨guard,axisOk,result,he,hm,hs,hd,ht,output,bytes,frame⟩ := checked_forward_graph
    x y dim dtype admitted allocated ax hnorm sms waves ops scalar R S memory before
  exact ⟨guard,axisOk,Binding.input_loads R "x" x memory before inputBytes,result,he,hm,hs,hd,ht,output,bytes,frame,
    Binding.storage_frame "out" y yStorage xStorage yWithin fresh before result.words,
    Binding.input_word_frame "out" "x" y x yStorage xStorage yWithin xWithin fresh before result.words memory inputBytes⟩

/-- Backward binds the actual saved-output Tensor and upstream Tensor, both
from the same before bytes. The original x is only the shape/device reference.
Read/read storage aliasing is allowed; only the fresh gradient is separated. -/
specification checked_backward (x saved grad y : VeriTile.Torch.TensorMetadata) (dim : Int)
    (dtype : Option VeriTile.Torch.DType) (admitted : ForwardContract x dim dtype)
    (savedMetadata : Allocated x saved) (gradient : GradientContract x grad) (allocated : Allocated x y)
    (ax : Fin x.shape.length) (hnorm : normalizeDim x.shape dim=some ax) (choice : Fin 6)
    (ops : NativeProfile.Operations) (scalar : ℝ → Word) (R : Reductions) (S : Schedule)
    (memory : RegionName → Nat → Word) (before : Binding.Bytes)
    (savedStorage gradStorage yStorage : Binding.Backing)
    (savedWithin : Binding.Within saved savedStorage) (gradWithin : Binding.Within grad gradStorage)
    (yWithin : Binding.Within y yStorage)
    (freshSaved : 0<y.numel → Binding.Separate yStorage savedStorage)
    (freshGrad : 0<y.numel → Binding.Separate yStorage gradStorage)
    (savedBytes : Binding.InputBytes "out" saved memory before) (gradBytes : Binding.InputBytes "grad" grad memory before) :
    Spec.Real (checkForward x dim dtype=.ok () ∧ checkGradient x grad=.ok () ∧ normalizeDim x.shape dim=some ax ∧
      saved.numel=x.numel ∧ grad.numel=x.numel ∧
      Binding.InputLoads R "out" saved memory before ∧ Binding.InputLoads R "grad" grad memory before ∧
      ∃ result, Tensor.backward ops scalar R S x y ax "out" "grad" "in_grad" (backwardConfig (dimensions x.shape ax) choice) memory=some result ∧
        result.tensor=y ∧ result.tensor.shape=x.shape ∧ result.tensor.device=x.device ∧ result.tensor.dtype=.floating .fp32 ∧
        let expected := Tensor.backwardWords ops scalar R S "out" "grad" "in_grad" (dimensions x.shape ax) (backwardConfig (dimensions x.shape ax) choice) memory
        let post := Binding.callerBytes "in_grad" y before result.words
        Tensor.Output "in_grad" x.numel expected memory result ∧
        (∀ i b, i<x.numel → b<4 → post (y.dataPtr.toNat+4*i+b)=Binding.byteOf (expected i) b) ∧
        (∀ a, a<y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤a → post a=before a) ∧
        (∀ a, savedStorage.base≤a ∧ a<savedStorage.base+savedStorage.bytes → post a=before a) ∧
        (∀ a, gradStorage.base≤a ∧ a<gradStorage.base+gradStorage.bytes → post a=before a) ∧
        (∀ i, i<saved.numel → Binding.wordAt post (saved.dataPtr.toNat+4*i)=memory "out" i) ∧
        (∀ i, i<grad.numel → Binding.wordAt post (grad.dataPtr.toNat+4*i)=memory "grad" i)) := by
  obtain ⟨guard,gradientOk,axisOk,result,he,hm,hs,hd,ht,output,bytes,frame⟩ := checked_backward_graph
    x grad y dim dtype admitted gradient allocated ax hnorm choice ops scalar R S memory before
  exact ⟨guard,gradientOk,axisOk,congrArg List.prod savedMetadata.1,congrArg List.prod gradient.2.1,Binding.input_loads R "out" saved memory before savedBytes,
    Binding.input_loads R "grad" grad memory before gradBytes,result,he,hm,hs,hd,ht,output,bytes,frame,
    Binding.storage_frame "in_grad" y yStorage savedStorage yWithin freshSaved before result.words,
    Binding.storage_frame "in_grad" y yStorage gradStorage yWithin freshGrad before result.words,
    Binding.input_word_frame "in_grad" "out" y saved yStorage savedStorage yWithin savedWithin freshSaved before result.words memory savedBytes,
    Binding.input_word_frame "in_grad" "grad" y grad yStorage gradStorage yWithin gradWithin freshGrad before result.words memory gradBytes⟩

#axiomsClean checked_forward
#axiomsClean checked_backward
#axiomsClean checked_forward_graph
#axiomsClean checked_backward_graph
#axiomsClean tensor_word_handoff
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.FPEquiv
