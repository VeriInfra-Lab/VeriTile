import bench.repairs.tritonbench_g.triton_matmul.Proofs.FiniteHost
import bench.repairs.tritonbench_g.triton_matmul.Proofs.Real
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.IdealTensor
open VeriTile.Triton FP.Structural FPValue
open Contract TensorBinding InputAdapter
noncomputable section

/-- Arbitrary real values indexed by physical element-start address. This is an
ideal interpretation of memory, with no finite encoding or decode restriction. -/
abbrev RealRead := Nat → ℝ

def logicalValue (read : RealRead) (t : Tensor) (i : Nat) : ℝ :=
  read (physicalAddress (storage t.metadata) t i)

structure Inputs (read : RealRead) (a b : Tensor) (s : State ℝ) : Prop where
  left : ∀ i,i<a.metadata.numel → s.mem TensorBinding.inputA i=
    stored (storage a.metadata) (read (a.metadata.dataPtr.toNat+width (storage a.metadata)*i))
  right : ∀ i,i<b.metadata.numel → s.mem TensorBinding.inputB i=
    stored (storage b.metadata) (read (b.metadata.dataPtr.toNat+width (storage b.metadata)*i))

theorem direct_read (read : RealRead) (t : Tensor) (laws : Laws (storage t.metadata) t)
    (contiguous : t.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<t.metadata.numel) :
    read (t.metadata.dataPtr.toNat+width (storage t.metadata)*i)=logicalValue read t i := by
  rw [laws.pointer (by omega)]
  simp only [logicalValue,physicalAddress,laws.contiguous contiguous i hi,Nat.mul_add,Nat.add_assoc]

theorem allocated_inputs (read : RealRead) (a b c : Tensor) (before prepared : State ℝ)
    (allocation : Allocation a b c before prepared) (bound : Inputs read a b before) :
    Inputs read a b prepared where
  left := fun i hi => (input_frame a b c before prepared allocation).1 i |>.trans (bound.left i hi)
  right := fun i hi => (input_frame a b c before prepared allocation).2 i |>.trans (bound.right i hi)

def tensorAPI (read : RealRead)
    (a b c : Tensor) (h : InputContract a.metadata b.metadata) (nonempty : ¬Empty a.metadata b.metadata)
    (before prepared : State ℝ) (allocation : Allocation a b c before prepared)
    (bound : Inputs read a b before) : TensorAPI (request a b) prepared := by
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
  have bound := allocated_inputs read a b c before prepared allocation bound
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


def left (read : RealRead) (a b : Tensor) : Fin (rows a.metadata) → Fin (columns a.metadata) → ℝ :=
  fun r k => logicalValue read a (r.val*columns a.metadata+k.val)
def right (read : RealRead) (a b : Tensor) : Fin (columns a.metadata) → Fin (columns b.metadata) → ℝ :=
  fun k c => logicalValue read b (k.val*columns b.metadata+c.val)

theorem source_inputs (read : RealRead) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (s : State ℝ) (bound : Inputs read a b s) :
    (∀ r k,Repair.inputA (request a b) s r.val k.val=left read a b r k) ∧
    (∀ k c,Repair.inputB (request a b) s k.val c.val=right read a b k c) := by
  constructor
  · intro r k
    have hi : r.val*columns a.metadata+k.val<a.metadata.numel := by
      rw [matrix_numel a.metadata h.rank.1]
      exact linear_bound _ _ _ _ r.isLt k.isLt
    simp only [Repair.inputA,request,Nat.mul_one,left]
    rw [bound.left _ hi,direct_read read a la h.contiguous.1 _ hi,read_stored]
  · intro k c
    have hi : k.val*columns b.metadata+c.val<b.metadata.numel := by
      rw [matrix_numel b.metadata h.rank.2,←h.inner]
      exact linear_bound _ _ _ _ k.isLt c.isLt
    simp only [Repair.inputB,request,Nat.mul_one,right]
    rw [bound.right _ hi,direct_read read b lb h.contiguous.2 _ hi,
      ←same_storage a.metadata b.metadata h.dtype.2,read_stored]

/-- Independent full-K mathematical target on unrestricted real logical inputs. -/
def expected (read : RealRead) (a b : Tensor) (i : OutputIndex a b) : ℝ :=
  ∑ k : Fin (columns a.metadata),left read a b i.1 k*right read a b k i.2

theorem reference (read : RealRead) (a b : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (s : State ℝ) (bound : Inputs read a b s) (i : OutputIndex a b) :
    logicalReference idealNumerics (request a b) (left read a b) (right read a b) i.1 i.2=expected read a b i := by
  have input := source_inputs read a b la lb h s bound
  rw [←reference_logical idealNumerics (request a b) s _ _ input.1 input.2 i.1 i.2,
    ideal_reference (request a b) s _ _ input.1 input.2 i.1 i.2]
  rfl

#specNonCircular expected avoiding [FiniteHost.call,CheckedHost.call,Source.typed,FP.Structural.exec]
#axiomsClean direct_read
#axiomsClean tensorAPI
#axiomsClean source_inputs
#axiomsClean reference
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.IdealTensor
