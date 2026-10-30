import bench.repairs.tritonbench_g.cosine_compute.Proofs.Native

/-! Ordinary Tensor/empty_like contracts plus actual source execution. Tensor
laws and allocator laws contain no cosine formula or future kernel result. -/
namespace VeriTile.Bench.Repairs.CosineCompute.Host
open VeriTile.Triton VeriTile.Torch.FP32
noncomputable section
set_option maxHeartbeats 6000000

abbrev Memory := Native.Memory
def slot (t : Tensor) (laws : TensorLaws t) (hc : t.metadata.isContiguous=Bool.true) : Nat :=
  (dataView t laws hc).base/4
def logical (s : Memory) (t : Tensor) (laws : TensorLaws t) (i : Nat) : Word :=
  logicalValue (fun byte=>s (byte/4)) t laws i

theorem slot_capacity (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous=Bool.true) : slot t laws hc+t.metadata.numel ≤ 2^62 := by
  have span := (dataView t laws hc).spanFits
  change (dataView t laws hc).base+4*t.metadata.numel ≤ 2^64 at span
  unfold slot
  omega

theorem logical_slot (s : Memory) (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous=Bool.true) (hn : t.metadata.isNeg=Bool.false)
    (i : Nat) (hi : i < t.metadata.numel) :
    logical s t laws i=s (slot t laws hc+i) := by
  rw [logical,←direct_read_eq_logical (fun byte=>s (byte/4)) t laws hc hn i hi]
  congr 1
  simp only [DataView.address,slot]
  omega

/-- Successful fresh empty_like allocation in the input's device. Its initial
output words are arbitrary. Frame and disjointness are allocator properties;
neither assumes a kernel result. -/
structure EmptyLikeResult (x : Tensor) (lx : TensorLaws x)
    (h : InputContract x.metadata) (before : Memory) where
  output : Tensor
  laws : TensorLaws output
  metadata : output.metadata={x.metadata with isNeg:=Bool.false,dataPtr:=output.dataPtr}
  allocated : Memory
  aligned : 0 < output.metadata.numel → output.dataPtr.toNat%4=0
  objectFresh : output.object ≠ x.object
  separation : ∀ i j, i < output.metadata.numel → j < x.metadata.numel →
    slot output laws (by rw [metadata]; exact h.contiguous)+i ≠ slot x lx h.contiguous+j
  frame : ∀ cell, (¬∃ i, i < output.metadata.numel ∧
    cell=slot output laws (by rw [metadata]; exact h.contiguous)+i) → allocated cell=before cell

theorem EmptyLikeResult.contiguous (a : EmptyLikeResult x lx h before) :
    a.output.metadata.isContiguous=Bool.true := by rw [a.metadata];exact h.contiguous
theorem EmptyLikeResult.numel (a : EmptyLikeResult x lx h before) :
    a.output.metadata.numel=x.metadata.numel := by
  simp [a.metadata,VeriTile.Torch.TensorMetadata.numel]

def layout (a : EmptyLikeResult x lx h before) : Native.Layout :=
  ⟨slot x lx h.contiguous,slot a.output a.laws a.contiguous,x.metadata.numel,256⟩

theorem allocation (a : EmptyLikeResult x lx h before) : Native.Allocated (layout a) := by
  refine ⟨slot_capacity x lx h.contiguous,?_,Or.inr ?_⟩
  · simpa only [layout,←a.numel] using slot_capacity a.output a.laws a.contiguous
  · intro i j hi hj
    exact a.separation i j (by rwa [a.numel]) hj

/-- The launch runs in the device selected by the source's CUDA context.
This ordinary successful-runtime fact has no value or output predicate. -/
structure Runtime (x : Tensor) where
  launchDevice : Nat
  selected : launchDevice=x.metadata.device.index

/-- Host order after successful allocation: checked arguments, empty return,
otherwise the exact finite source at a complete grid, returning this allocation. -/
def runWrapper (b : CosineNative.Numerics) (x : Tensor) (lx : TensorLaws x)
    (h : InputContract x.metadata) (before : Memory) (a : EmptyLikeResult x lx h before)
    (order : List Nat) : Except InputError (Tensor × Memory) := do
  checkTensorInput x.metadata
  if x.metadata.numel=0 then return (a.output,a.allocated)
  return (a.output,Native.execute b (layout a) order a.allocated)

theorem wrapper_correct (b : CosineNative.Numerics) (x : Tensor) (lx : TensorLaws x)
    (h : InputContract x.metadata) (before : Memory) (a : EmptyLikeResult x lx h before)
    (order : List Nat) (complete : order.Perm (List.range (gridCount x.metadata.numel))) :
    ∃ final, runWrapper b x lx h before a order = .ok (a.output,final) ∧
      (∀ i, i < x.metadata.numel → logical final a.output a.laws i=Native.scalar b (logical before x lx i)) ∧
      (∀ cell, (¬∃ i, i < x.metadata.numel ∧ cell=(layout a).output+i) → final cell=before cell) := by
  have accepted := (checkTensorInput_iff x.metadata).mpr h
  by_cases empty : x.metadata.numel=0
  · refine ⟨a.allocated,by simp [runWrapper,accepted,empty],by simp [empty],?_⟩
    intro cell outside
    exact a.frame cell (by simpa only [a.numel,layout] using outside)
  have result := Native.launch_correct b (layout a) (allocation a) rfl h.extent order complete a.allocated
  refine ⟨Native.execute b (layout a) order a.allocated,by simp [runWrapper,accepted,empty],?_,?_⟩
  · intro i hi
    rw [logical_slot _ a.output a.laws a.contiguous (by rw [a.metadata]) i (by rwa [a.numel]),
      logical_slot before x lx h.contiguous h.noNeg i hi]
    change Native.execute b (layout a) order a.allocated ((layout a).output+i)=_
    rw [result.1 i hi]
    apply congrArg (Native.scalar b)
    apply a.frame
    rintro ⟨j,hj,eq⟩
    exact a.separation j i hj hi eq.symm
  · intro cell outside
    rw [result.2 cell outside]
    exact a.frame cell (by simpa only [a.numel,layout] using outside)

theorem pointer_binding (x : Tensor) (lx : TensorLaws x) (h : InputContract x.metadata)
    (before : Memory) (a : EmptyLikeResult x lx h before) (nonempty : 0 < x.metadata.numel) :
    BitVec.ofNat 64 (4*(layout a).input)=x.dataPtr ∧
    BitVec.ofNat 64 (4*(layout a).output)=a.output.dataPtr := by
  have provePointer (t : Tensor) (lt : TensorLaws t) (hc : t.metadata.isContiguous=Bool.true)
      (ha : 0 < t.metadata.numel → t.dataPtr.toNat%4=0) (hp : 0 < t.metadata.numel) :
      BitVec.ofNat 64 (4*slot t lt hc)=t.dataPtr := by
    have aligned := dataView_aligned t lt hc ha
    have exactBase : 4*slot t lt hc=(dataView t lt hc).base := by unfold slot;omega
    rw [exactBase]
    simpa [DataView.address] using actual_pointer t lt hc hp
  exact ⟨provePointer x lx h.contiguous h.alignment nonempty,
    provePointer a.output a.laws a.contiguous a.aligned (by rwa [a.numel])⟩

#axiomsClean allocation
#axiomsClean wrapper_correct
#axiomsClean pointer_binding
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CosineCompute.Host
