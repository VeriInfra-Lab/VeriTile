import bench.repairs.tritonbench_g.softmax_flaggems.Kernels
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.NativeProfile
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Binding
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Pointers
import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Inputs

namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section

def layouts (emptyLike : List Nat → List Nat → List Nat) : LayoutPrimitives :=
  ⟨fun _ strides => strides,emptyLike⟩

/-- Independent complete expression graph, including all tuning trials and
saved-output loads. The graph preserves source FP32 casts and load barriers. -/
abbrev reference := FPHost.reference (α := Native.Term)

/-- The same public guard domain as the Real theorem, with complete source
execution, saved state, metadata, and instruction-level FP32 readback. -/
specification softmax_autograd_words (x grad : VeriTile.Torch.TensorMetadata)
    (dim : Int) (dtype : Option VeriTile.Torch.DType)
    (admitted : ForwardContract x dim dtype) (gradient : GradientContract x grad)
    (ops : NativeProfile.G.Operations) (scalar : ℝ → Native.Word)
    (R : Native.Reductions) (S : Native.Schedule)
    (sms : Nat) (waves : WavesAboveOne) (emptyLike : List Nat → List Nat → List Nat)
    (inputStrides : List Nat) (trials : List (Fin 6)) (choice : Fin 6)
    (memory : RegionName → Nat → Native.Word)
    (inputValues emptyOutput gradValues emptyGradient : Nat → Native.Word) :
    Spec.Real (checkForward x dim dtype=.ok () ∧ checkGradient x grad=.ok () ∧
      ∃ (ax : Fin x.shape.length) (result : FPHost.Result Native.Term),
        normalizeDim x.shape dim=some ax ∧
        FPHost.invoke (Native.algebra R) x.shape dim x.device.index sms waves (layouts emptyLike)
          inputStrides trials choice "x" "out" "grad" "in_grad" (Native.entry memory)
          (Native.input "x" inputValues) (Native.input "out" emptyOutput)
          (Native.input "grad" gradValues) (Native.input "in_grad" emptyGradient)=.ok result ∧
        let expected := reference (Native.algebra R) x.shape ax x.device.index sms waves (layouts emptyLike)
          inputStrides trials choice "x" "out" "grad" "in_grad" (Native.entry memory)
          (Native.input "x" inputValues) (Native.input "out" emptyOutput)
          (Native.input "grad" gradValues) (Native.input "in_grad" emptyGradient)
        FPHost.Matches result expected ∧
        FPHost.Saved "out" result ∧
        FPHost.HostFrame "x" "out" "grad" "in_grad" x.numel (Native.entry memory) result ∧
        (∀ o, o<x.numel → Native.observe (NativeProfile.instructions ops scalar) S result.forwardState "out" o=
          NativeProfile.G.eval ops (NativeProfile.lower scalar S ((expected.forwardState.mem "out" o).read .real))) ∧
        (∀ o, o<x.numel → Native.observe (NativeProfile.instructions ops scalar) S result.finalState "in_grad" o=
          NativeProfile.G.eval ops (NativeProfile.lower scalar S ((expected.finalState.mem "in_grad" o).read .real)))) := by
  obtain ⟨ax,hnorm⟩ := admitted_axis x dim dtype admitted
  obtain ⟨result,he,hm,hs,hf⟩ := FPHost.whole_typed (Native.algebra R) x.shape dim ax hnorm
    x.device.index sms waves (layouts emptyLike) inputStrides trials choice
    "x" "out" "grad" "in_grad" (Native.entry memory) (Native.input "x" inputValues)
    (Native.input "out" emptyOutput) (Native.input "grad" gradValues) (Native.input "in_grad" emptyGradient) (by decide)
  refine ⟨(checkForward_iff x dim dtype).mpr admitted,(checkGradient_iff x grad).mpr gradient,
    ax,result,hnorm,he,hm,hs,hf,?_,?_⟩
  · intro o ho
    unfold Native.observe
    rw [hm.1]
    exact NativeProfile.lower_exact ops scalar S _
  · intro o ho
    unfold Native.observe
    rw [hm.2.1]
    exact NativeProfile.lower_exact ops scalar S _

/-- Ordinary allocated output descriptor: metadata, live pointer span and
alignment. It contains no value, kernel result, or numerical condition. -/
def Allocated (x p : VeriTile.Torch.TensorMetadata) : Prop :=
  p.shape=x.shape ∧ p.device=x.device ∧ p.dtype=.floating .fp32 ∧
  p.layout=.strided ∧ p.isContiguous=Bool.true ∧ p.isNeg=Bool.false ∧
  (0<p.numel → p.dataPtr.toNat%4=0 ∧ p.dataPtr.toNat+4*p.numel≤2^64)

/-- Four-byte readback is tied to the allocated output's observed data pointer.
Its frame preserves every byte outside that region. -/
theorem result_bytes (ops : NativeProfile.G.Operations) (scalar : ℝ → Native.Word)
    (S : Native.Schedule) (x p y : VeriTile.Torch.TensorMetadata)
    (hp : Allocated x p) (hy : Allocated x y)
    (actual expected : FPHost.Result Native.Term) (hm : FPHost.Matches actual expected)
    (before : Binding.Bytes) :
    (∀ i b, i<x.numel → b<4 →
      Binding.callerBytes (NativeProfile.instructions ops scalar) S "out" p before actual.forwardState
        (p.dataPtr.toNat+4*i+b)=Binding.byteOf
          (NativeProfile.G.eval ops (NativeProfile.lower scalar S ((expected.forwardState.mem "out" i).read .real))) b) ∧
    (∀ i b, i<x.numel → b<4 →
      Binding.callerBytes (NativeProfile.instructions ops scalar) S "in_grad" y before actual.finalState
        (y.dataPtr.toNat+4*i+b)=Binding.byteOf
          (NativeProfile.G.eval ops (NativeProfile.lower scalar S ((expected.finalState.mem "in_grad" i).read .real))) b) ∧
    (∀ address, address<p.dataPtr.toNat ∨ p.dataPtr.toNat+4*p.numel≤address →
      Binding.callerBytes (NativeProfile.instructions ops scalar) S "out" p before actual.forwardState address=before address) ∧
    (∀ address, address<y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤address →
      Binding.callerBytes (NativeProfile.instructions ops scalar) S "in_grad" y before actual.finalState address=before address) := by
  have ps : p.numel=x.numel := congrArg List.prod hp.1
  have ys : y.numel=x.numel := congrArg List.prod hy.1
  refine ⟨?_,?_,?_,?_⟩
  · intro i b hi hb
    rw [Binding.coordinate_bytes _ _ _ _ _ _ _ _ (by omega) hb,hm.1,NativeProfile.lower_exact]
  · intro i b hi hb
    rw [Binding.coordinate_bytes _ _ _ _ _ _ _ _ (by omega) hb,hm.2.1,NativeProfile.lower_exact]
  · exact Binding.exterior_bytes _ _ _ _ _ _
  · exact Binding.exterior_bytes _ _ _ _ _ _

/-- Whole-call byte result with direct input leaves. Allocated descriptors are
bound to the two source allocator results; `before` is their surrounding memory.
Source store coverage is derived by `Native.host_native_coverage`. -/
specification softmax_autograd_bytes (x grad p y : VeriTile.Torch.TensorMetadata)
    (dim : Int) (dtype : Option VeriTile.Torch.DType)
    (admitted : ForwardContract x dim dtype) (gradient : GradientContract x grad)
    (hp : Allocated x p) (hy : Allocated x y) (fresh : Binding.Separate p y)
    (ops : NativeProfile.G.Operations) (scalar : ℝ → Native.Word)
    (R : Native.Reductions) (S : Native.Schedule)
    (sms : Nat) (waves : WavesAboveOne) (emptyLike : List Nat → List Nat → List Nat)
    (inputStrides : List Nat) (trials : List (Fin 6)) (choice : Fin 6)
    (memory : RegionName → Nat → Native.Word) (before : Binding.Bytes)
    (emptyOutput emptyGradient : Nat → Native.Word) :
    Spec.Real (∃ (ax : Fin x.shape.length) (result : FPHost.Result Native.Term),
      normalizeDim x.shape dim=some ax ∧
      FPHost.invoke (Native.algebra R) x.shape dim x.device.index sms waves (layouts emptyLike)
        inputStrides trials choice "x" "out" "grad" "in_grad" (Native.entry memory)
        (Native.input "x" (memory "x")) (Native.input "out" emptyOutput)
        (Native.input "grad" (memory "grad")) (Native.input "in_grad" emptyGradient)=.ok result ∧
      let expected := reference (Native.algebra R) x.shape ax x.device.index sms waves (layouts emptyLike)
        inputStrides trials choice "x" "out" "grad" "in_grad" (Native.entry memory)
        (Native.input "x" (memory "x")) (Native.input "out" emptyOutput)
        (Native.input "grad" (memory "grad")) (Native.input "in_grad" emptyGradient)
      let bytes := Binding.returnedBytes (NativeProfile.instructions ops scalar) S p y before
        result.forwardState result.finalState
      FPHost.Matches result expected ∧ FPHost.Saved "out" result ∧
      (∀ i b, i<x.numel → b<4 → bytes (p.dataPtr.toNat+4*i+b)=Binding.byteOf
        (NativeProfile.G.eval ops (NativeProfile.lower scalar S ((expected.forwardState.mem "out" i).read .real))) b) ∧
      (∀ i b, i<x.numel → b<4 → bytes (y.dataPtr.toNat+4*i+b)=Binding.byteOf
        (NativeProfile.G.eval ops (NativeProfile.lower scalar S ((expected.finalState.mem "in_grad" i).read .real))) b) ∧
      (∀ address, (address<p.dataPtr.toNat ∨ p.dataPtr.toNat+4*p.numel≤address) →
        (address<y.dataPtr.toNat ∨ y.dataPtr.toNat+4*y.numel≤address) → bytes address=before address) ∧
      (∀ (region : RegionName) o, ¬((region="out" ∨ region="in_grad") ∧ o<x.numel) →
        result.finalState.mem region o=(Native.entry memory).mem region o)) := by
  obtain ⟨_,_,ax,result,hnorm,he,hm,hs,hf,hw,hg⟩ := softmax_autograd_words x grad dim dtype admitted gradient
    ops scalar R S sms waves emptyLike inputStrides trials choice memory
    (memory "x") emptyOutput (memory "grad") emptyGradient
  have ps : p.numel=x.numel := congrArg List.prod hp.1
  have ys : y.numel=x.numel := congrArg List.prod hy.1
  refine ⟨ax,result,hnorm,he,hm,hs,?_,?_,?_,?_⟩
  · intro i b hi hb
    rw [Binding.returned_output _ _ _ _ fresh _ _ _ _ _ (by omega) hb]
    exact congrArg (fun w => Binding.byteOf w b) (hw i hi)
  · intro i b hi hb
    rw [Binding.returned_gradient _ _ _ _ _ _ _ _ _ (by omega) hb]
    exact congrArg (fun w => Binding.byteOf w b) (hg i hi)
  · exact Binding.returned_frame _ _ _ _ _ _ _
  · intro region o outside
    rw [hm.2.1]
    exact Inputs.reference_frame R x.shape ax x.device.index sms waves (layouts emptyLike) inputStrides
      trials choice memory emptyOutput emptyGradient region o outside

/-- Conditional numerical composition inspects each actual intermediate
instruction operand. It does not restrict the wrapper's structural theorem. -/
theorem output_exact_if_local (ops : NativeProfile.G.Operations) (scalar : ℝ → Native.Word)
    (S : Native.Schedule) (ref : Ksoftmax.Words.PrimitiveReference)
    (ports : NativeProfile.LocalExact ops ref) (actual expected : FPHost.Result Native.Term)
    (hm : FPHost.Matches actual expected) (o : Nat)
    (domain : NativeProfile.NumericDomain ops
      (NativeProfile.lower scalar S ((expected.forwardState.mem "out" o).read .real))) :
    Native.observe (NativeProfile.instructions ops scalar) S actual.forwardState "out" o=
      NativeProfile.G.eval (Ksoftmax.Words.referenceOps ref)
        (NativeProfile.lower scalar S ((expected.forwardState.mem "out" o).read .real)) := by
  unfold Native.observe
  rw [hm.1]
  exact NativeProfile.exact_if_local ops scalar S ref ports _ domain

theorem gradient_exact_if_local (ops : NativeProfile.G.Operations) (scalar : ℝ → Native.Word)
    (S : Native.Schedule) (ref : Ksoftmax.Words.PrimitiveReference)
    (ports : NativeProfile.LocalExact ops ref) (actual expected : FPHost.Result Native.Term)
    (hm : FPHost.Matches actual expected) (o : Nat)
    (domain : NativeProfile.NumericDomain ops
      (NativeProfile.lower scalar S ((expected.finalState.mem "in_grad" o).read .real))) :
    Native.observe (NativeProfile.instructions ops scalar) S actual.finalState "in_grad" o=
      NativeProfile.G.eval (Ksoftmax.Words.referenceOps ref)
        (NativeProfile.lower scalar S ((expected.finalState.mem "in_grad" o).read .real)) := by
  unfold Native.observe
  rw [hm.2.1]
  exact NativeProfile.exact_if_local ops scalar S ref ports _ domain

#specNonCircular reference avoiding [FPHost.invoke,FPHost.nonempty,FPGrid.run,FP.Structural.exec]
#specNonCircular Allocated avoiding [FPHost.invoke,FPHost.reference]
#axiomsClean softmax_autograd_words
#axiomsClean result_bytes
#axiomsClean softmax_autograd_bytes
#axiomsClean output_exact_if_local
#axiomsClean gradient_exact_if_local
end
end VeriTile.Bench.Repairs.SoftmaxFlaggems.FPEquiv
