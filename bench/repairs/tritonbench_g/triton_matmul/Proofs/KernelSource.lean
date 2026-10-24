import bench.repairs.tritonbench_g.triton_matmul.KernelContract
import bench.repairs.tritonbench_g.triton_matmul.Proofs.FiniteGrid
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelSource
open VeriTile.Triton FP.Structural FPValue
open Contract TensorBinding InputAdapter KernelContract
noncomputable section
set_option maxHeartbeats 1000000

/-- Derive the finite machine ABI directly from admitted inputs and the existing
output tensor. Singleton unused strides are canonicalized exactly as Python does. -/
def tensorAPI {α : Type} [Inhabited α] (read : Storage → Nat → α)
    (a b c : Tensor) (h : InputContract a.metadata b.metadata) (nonempty : ¬Empty a.metadata b.metadata)
    (prepared : State α) (output : Output a b c)
    (bound : KernelContract.Inputs read a b prepared) : TensorAPI (request a b) prepared := by
  have numA := matrix_numel a.metadata h.rank.1
  have numB := matrix_numel b.metadata h.rank.2
  have numC := output_numel a b c output.metadata
  have hs := same_storage a.metadata b.metadata h.dtype.2
  have ha := pointer_window a.metadata (execution_nonempty _ _ h nonempty).2.1
  have hb := pointer_window b.metadata (execution_nonempty _ _ h nonempty).2.2
  have hc := pointer_window c.metadata output.window
  have ew := execution_nonempty _ _ h nonempty
  have ba := checked_bytes a.metadata
  have bb : bytes b.metadata=width (storage a.metadata) := by rw [checked_bytes,hs]
  have bc := output_bytes a b c output.metadata
  have elem : elementBytes (storage a.metadata)=width (storage a.metadata) := by cases storage a.metadata <;> rfl
  have freshA : MatmulTritonAutotune.Whole.byteSpansDisjoint (base c)
      (width (storage a.metadata)*c.metadata.numel) (base a) (width (storage a.metadata)*a.metadata.numel) := by
    by_cases ec : c.metadata.numel=0
    · simp [MatmulTritonAutotune.Whole.byteSpansDisjoint,ec]
    by_cases ea : a.metadata.numel=0
    · simp [MatmulTritonAutotune.Whole.byteSpansDisjoint,ea]
    simpa only [base,ec,ea,↓reduceIte] using output.separateA
  have freshB : MatmulTritonAutotune.Whole.byteSpansDisjoint (base c)
      (width (storage a.metadata)*c.metadata.numel) (base b) (width (storage a.metadata)*b.metadata.numel) := by
    by_cases ec : c.metadata.numel=0
    · simp [MatmulTritonAutotune.Whole.byteSpansDisjoint,ec]
    by_cases eb : b.metadata.numel=0
    · simp [MatmulTritonAutotune.Whole.byteSpansDisjoint,eb]
    simpa only [base,ec,eb,↓reduceIte] using output.separateB
  refine {
    extentA := a.metadata.numel, extentB := b.metadata.numel
    baseA := base a, baseB := base b, baseC := base c
    metadataA := ⟨[rows a.metadata,columns a.metadata],[columns a.metadata,1],dtype (storage a.metadata),a.metadata.device.kind⟩
    metadataB := ⟨[rows b.metadata,columns b.metadata],[columns b.metadata,1],dtype (storage a.metadata),b.metadata.device.kind⟩
    metadataC := ⟨[rows a.metadata,columns b.metadata],[max (columns b.metadata) 1,1],dtype (storage a.metadata),c.metadata.device.kind⟩
    shapeA := rfl, shapeB := rfl, shapeC := rfl, stridesA := rfl, stridesB := rfl, stridesC := rfl
    dtypeA := rfl, dtypeB := rfl, dtypeC := rfl
    deviceB := congrArg (fun d => d.kind) h.device.2.2
    deviceC := congrArg (fun d => d.kind) output.metadata.2.2.1
    dimensions := h.inner, storage := hs.symm
    dimM := h.dimensions.1, dimN := h.dimensions.2.1, dimK := h.dimensions.2.2
    freshA := by simp [request,outputC,TensorBinding.inputA]
    freshB := by simp [request,outputC,TensorBinding.inputB]
    physicalFreshA := ?_, physicalFreshB := ?_
    alignmentA := ?_, alignmentB := ?_, alignmentC := ?_
    spanA := ?_, spanB := ?_, spanC := ?_
    allocationA := ?_, allocationB := ?_, allocationC := ?_
    strideAM := h.dimensions.2.2, strideAK := by simp [request]
    strideBK := h.dimensions.2.1, strideBN := by simp [request]
    inputA := ?_, inputB := ?_, tagA := ?_, tagB := ?_
    sameViewOrigin := ?_, aliasConsistency := ?_ }
  · simpa only [request,elem,←numC] using freshA
  · simpa only [request,elem,←numC] using freshB
  · simpa only [request,elem,ba,base] using ha.1
  · simpa only [request,elem,bb,base] using hb.1
  · simpa only [request,elem,bc,base] using hc.1
  · simpa only [request,elem,ba,base] using ha.2
  · simpa only [request,elem,bb,base] using hb.2
  · simpa only [request,elem,bc,base,numC] using hc.2
  · simpa only [request,elem,numA,ba] using ew.1.1
  · simpa only [request,elem,numB,←h.inner,ba] using ew.1.2.1
  · simpa only [request,elem,ba] using ew.1.2.2.1
  · intro r hr k hk
    simp only [request,Nat.mul_one]
    rw [numA]
    exact linear_bound _ _ _ _ hr hk
  · intro k hk col hcol
    simp only [request,Nat.mul_one]
    rw [numB,←h.inner]
    exact linear_bound _ _ _ _ hk hcol
  · intro r hr k hk
    have hi : r*columns a.metadata+k<a.metadata.numel := by rw [numA]; exact linear_bound _ _ _ _ hr hk
    exact ⟨_,by simpa only [request,Nat.mul_one] using bound.left _ hi⟩
  · intro k hk col hcol
    have hi : k*columns b.metadata+col<b.metadata.numel := by rw [numB,←h.inner]; exact linear_bound _ _ _ _ hk hcol
    exact ⟨_,by simpa only [request,Nat.mul_one,hs] using bound.right _ hi⟩
  · simp [request,TensorBinding.inputA,TensorBinding.inputB]
  · intro r hr k hk k' hk' col hcol he
    have ai : r*columns a.metadata+k<a.metadata.numel := by rw [numA]; exact linear_bound _ _ _ _ hr hk
    have bi : k'*columns b.metadata+col<b.metadata.numel := by rw [numB,←h.inner]; exact linear_bound _ _ _ _ hk' hcol
    have ae : a.metadata.numel≠0 := by omega
    have be : b.metadata.numel≠0 := by omega
    simp only [request,Nat.mul_one,base,ae,be,↓reduceIte,elem] at he ⊢
    rw [bound.left _ ai,bound.right _ bi,hs,he]


#axiomsClean tensorAPI
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelSource
