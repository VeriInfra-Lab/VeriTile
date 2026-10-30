import bench.repairs.tritonbench_g.cosine_compute.Proofs.Host

namespace VeriTile.Bench.Repairs.CosineCompute.FPEquiv
open VeriTile.Triton VeriTile.Torch.FP32
noncomputable section

/-- The complete finite source returns the actual empty_like Tensor. Every
FP32 bit pattern remains admitted; tl.cos is the unchanged opaque primitive.
The independent result is one primitive at each entry-state logical input. -/
specification cosine_wrapper_fp32 (b : CosineNative.Numerics)
    (x : Tensor) (lx : TensorLaws x) (h : InputContract x.metadata)
    (before : Host.Memory) (a : Host.EmptyLikeResult x lx h before)
    (runtime : Host.Runtime x) (order : List Nat)
    (complete : order.Perm (List.range (gridCount x.metadata.numel))) :
    ∃ final, Host.runWrapper b x lx h before a order = .ok (a.output,final) ∧
      (∀ i, i < x.metadata.numel → Host.logical final a.output a.laws i =
        Native.scalar b (Host.logical before x lx i)) ∧
      (∀ cell, (¬∃ i, i < x.metadata.numel ∧ cell=(Host.layout a).output+i) →
        final cell=before cell) ∧
      (0 < x.metadata.numel →
        BitVec.ofNat 64 (4*(Host.layout a).input)=x.dataPtr ∧
        BitVec.ofNat 64 (4*(Host.layout a).output)=a.output.dataPtr) ∧
      runtime.launchDevice=x.metadata.device.index ∧
      a.output.metadata.shape=x.metadata.shape ∧
      a.output.metadata.device=x.metadata.device ∧
      a.output.metadata.dtype=.floating .fp32 := by
  obtain ⟨final,he,hv,hf⟩ := Host.wrapper_correct b x lx h before a order complete
  exact ⟨final,he,hv,hf,Host.pointer_binding x lx h before a,runtime.selected,
    by simp [a.metadata],by simp [a.metadata],by simpa [a.metadata] using h.dtype⟩

/-- Proposed local numerical obligation for the benchmark's finite input
range. This is a hypothesis pending the shared GPU experiment, not an axiom
or a claim that Triton implements correctly-rounded mathematical cosine. -/
def ScalarAccuracy (input output : Word) : Prop :=
  ∀ q : Rat, input.toRat?=some q → |(q : ℝ)| ≤ 2*Real.pi →
    ∃ r : Rat, output.toRat?=some r ∧ |(r : ℝ)-Real.cos (q : ℝ)| ≤ (1 : ℝ)/1048576

def LocalCosAccuracy (b : CosineNative.Numerics) : Prop :=
  ∀ input : Word, ScalarAccuracy input (Native.scalar b input)

/-- Transport the local FP32 numerical premise through the complete finite
source and actual returned Tensor. Other input values retain the structural
result; the wrapper's checked domain is not narrowed. -/
specification cosine_accuracy_if_local (b : CosineNative.Numerics)
    (numerical : LocalCosAccuracy b) (x : Tensor) (lx : TensorLaws x)
    (h : InputContract x.metadata) (before : Host.Memory)
    (a : Host.EmptyLikeResult x lx h before) (order : List Nat)
    (complete : order.Perm (List.range (gridCount x.metadata.numel))) :
    ∃ final, Host.runWrapper b x lx h before a order = .ok (a.output,final) ∧
      ∀ i, i < x.metadata.numel →
        ScalarAccuracy (Host.logical before x lx i) (Host.logical final a.output a.laws i) := by
  obtain ⟨final,he,hv,_⟩ := Host.wrapper_correct b x lx h before a order complete
  refine ⟨final,he,?_⟩
  intro i hi
  rw [hv i hi]
  exact numerical _

#specNonCircular ScalarAccuracy avoiding [Host.runWrapper,Native.execute,Native.program]
#axiomsClean cosine_wrapper_fp32
#axiomsClean cosine_accuracy_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CosineCompute.FPEquiv
