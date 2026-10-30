import bench.repairs.tritonbench_g.triton_matmul.Proofs.TensorBinding
import bench.repairs.tritonbench_g.triton_matmul.Proofs.Index
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.InputAdapter
open VeriTile.Triton FP.Structural FPValue
open Contract TensorBinding
noncomputable section

def request (a b : Tensor) : Request :=
  ⟨TensorBinding.inputA,TensorBinding.inputB,outputC,storage a.metadata,storage b.metadata,
    rows a.metadata,columns b.metadata,columns a.metadata,rows b.metadata,
    columns a.metadata,1,columns b.metadata,1⟩
abbrev OutputIndex (a b : Tensor) := Fin (rows a.metadata) × Fin (columns b.metadata)
def address (a b : Tensor) (i : OutputIndex a b) := i.1.val*columns b.metadata+i.2.val

def Metadata (a b c : Tensor) : Prop :=
  c.metadata.shape=[rows a.metadata,columns b.metadata] ∧ c.metadata.dtype=a.metadata.dtype ∧
  c.metadata.device=a.metadata.device ∧ c.metadata.isContiguous=Bool.true ∧ c.metadata.isNeg=Bool.false

/-- A successful ordinary fresh output allocation. No source index envelope,
kernel result, matmul formula, or successful source execution is a field. -/
structure Allocation {α : Type} (a b c : Tensor) (before prepared : State α) : Prop where
  metadata : Metadata a b c
  objects : c.object≠a.object ∧ c.object≠b.object
  window : Contract.Window c.metadata
  separateA : MatmulTritonAutotune.Whole.byteSpansDisjoint c.metadata.dataPtr.toNat
    (width (storage a.metadata)*c.metadata.numel) a.metadata.dataPtr.toNat
    (width (storage a.metadata)*a.metadata.numel)
  separateB : MatmulTritonAutotune.Whole.byteSpansDisjoint c.metadata.dataPtr.toNat
    (width (storage a.metadata)*c.metadata.numel) b.metadata.dataPtr.toNat
    (width (storage a.metadata)*b.metadata.numel)
  frame : ∀ r o,(r≠outputC ∨ c.metadata.numel≤o) → prepared.mem r o=before.mem r o

abbrev Allocator (α : Type) := Tensor → Tensor → State α → Option (Tensor × State α)
def AllocationLaw {α : Type} (allocate : Allocator α) (a b : Tensor) (before : State α) : Prop :=
  ∃ c prepared,allocate a b before=some (c,prepared) ∧ Allocation a b c before prepared

theorem input_frame {α : Type} (a b c : Tensor) (before prepared : State α)
    (allocation : Allocation a b c before prepared) :
    (∀ i,prepared.mem TensorBinding.inputA i=before.mem TensorBinding.inputA i) ∧
    (∀ i,prepared.mem TensorBinding.inputB i=before.mem TensorBinding.inputB i) := by
  exact ⟨fun i => allocation.frame _ i (Or.inl (by simp [TensorBinding.inputA,outputC])),
    fun i => allocation.frame _ i (Or.inl (by simp [TensorBinding.inputB,outputC]))⟩

theorem allocated_inputs {α : Type} (read : Read) (decode : Word → α)
    (a b c : Tensor) (before prepared : State α) (allocation : Allocation a b c before prepared)
    (bound : Inputs read decode a b before) : Inputs read decode a b prepared where
  left := fun i hi => (input_frame a b c before prepared allocation).1 i |>.trans (bound.left i hi)
  right := fun i hi => (input_frame a b c before prepared allocation).2 i |>.trans (bound.right i hi)

theorem output_numel (a b c : Tensor) (metadata : Metadata a b c) :
    c.metadata.numel=rows a.metadata*columns b.metadata := by
  simp [VeriTile.Torch.TensorMetadata.numel,metadata.1]

theorem output_bytes (a b c : Tensor) (metadata : Metadata a b c) :
    bytes c.metadata=width (storage a.metadata) := by
  simp only [bytes,metadata.2.1]
  exact checked_bytes a.metadata

theorem pointer_window (t : VeriTile.Torch.TensorMetadata) (h : Window t) :
    (if t.numel=0 then 0 else t.dataPtr.toNat)%bytes t=0 ∧
    (if t.numel=0 then 0 else t.dataPtr.toNat)+bytes t*t.numel≤2^64 := by
  rcases h with empty | h
  · simp [empty]
  · by_cases empty : t.numel=0 <;> simp [empty,h.1,h.2]

def base (t : Tensor) := if t.metadata.numel=0 then 0 else t.metadata.dataPtr.toNat

/-- Build the historical finite-source interface from original pointer views,
explicit admission guards, and ordinary successful fresh allocation. Its old
allocation-size bounds are derived here from Contract.Envelope. -/
def tensorAPI {α : Type} [Inhabited α] (read : Read) (decode : Word → α)
    (a b c : Tensor) (h : InputContract a.metadata b.metadata) (nonempty : ¬Empty a.metadata b.metadata)
    (before prepared : State α) (allocation : Allocation a b c before prepared)
    (bound : Inputs read decode a b before) : TensorAPI (request a b) prepared := by
  have numA := matrix_numel a.metadata h.rank.1
  have numB := matrix_numel b.metadata h.rank.2
  have numC := output_numel a b c allocation.metadata
  have hs := same_storage a.metadata b.metadata h.dtype.2
  have ha := pointer_window a.metadata (execution_nonempty _ _ h nonempty).2.1
  have hb := pointer_window b.metadata (execution_nonempty _ _ h nonempty).2.2
  have hc := pointer_window c.metadata allocation.window
  have ew := execution_nonempty _ _ h nonempty
  have ba := checked_bytes a.metadata
  have bb : bytes b.metadata=width (storage a.metadata) := by rw [checked_bytes,hs]
  have bc := output_bytes a b c allocation.metadata
  have bound := allocated_inputs read decode a b c before prepared allocation bound
  have elem : elementBytes (storage a.metadata)=width (storage a.metadata) := by cases storage a.metadata <;> rfl
  have freshA : MatmulTritonAutotune.Whole.byteSpansDisjoint (base c)
      (width (storage a.metadata)*c.metadata.numel) (base a) (width (storage a.metadata)*a.metadata.numel) := by
    by_cases ec : c.metadata.numel=0
    · simp [MatmulTritonAutotune.Whole.byteSpansDisjoint,ec]
    by_cases ea : a.metadata.numel=0
    · simp [MatmulTritonAutotune.Whole.byteSpansDisjoint,ea]
    simpa only [base,ec,ea,↓reduceIte] using allocation.separateA
  have freshB : MatmulTritonAutotune.Whole.byteSpansDisjoint (base c)
      (width (storage a.metadata)*c.metadata.numel) (base b) (width (storage a.metadata)*b.metadata.numel) := by
    by_cases ec : c.metadata.numel=0
    · simp [MatmulTritonAutotune.Whole.byteSpansDisjoint,ec]
    by_cases eb : b.metadata.numel=0
    · simp [MatmulTritonAutotune.Whole.byteSpansDisjoint,eb]
    simpa only [base,ec,eb,↓reduceIte] using allocation.separateB
  refine {
    extentA := a.metadata.numel, extentB := b.metadata.numel
    baseA := base a, baseB := base b, baseC := base c
    metadataA := ⟨[rows a.metadata,columns a.metadata],[columns a.metadata,1],dtype (storage a.metadata),a.metadata.device.kind⟩
    metadataB := ⟨[rows b.metadata,columns b.metadata],[columns b.metadata,1],dtype (storage a.metadata),b.metadata.device.kind⟩
    metadataC := ⟨[rows a.metadata,columns b.metadata],[max (columns b.metadata) 1,1],dtype (storage a.metadata),c.metadata.device.kind⟩
    shapeA := rfl, shapeB := rfl, shapeC := rfl, stridesA := rfl, stridesB := rfl, stridesC := rfl
    dtypeA := rfl, dtypeB := rfl, dtypeC := rfl
    deviceB := congrArg (fun d => d.kind) h.device.2.2
    deviceC := congrArg (fun d => d.kind) allocation.metadata.2.2.1
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
#axiomsClean allocated_inputs
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.InputAdapter
