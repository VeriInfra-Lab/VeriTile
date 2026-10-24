import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Misc.Auxiliary
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.RealTensor

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.TensorBridge
noncomputable section

section Mathematical
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership RealTensor

/-- Complete actual candidate quotation, independent six-input Real target,
padded O/M/L writes and exterior cell frame. The native statement is separate. -/
theorem whole_real (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (args : Checked.Arguments) (admission : Contract.Admission c A p args)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (input : BlockState) (defined : input.undef=(fun _ _=>0)) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
      (A.flattenKernel (Kernels.source c).eraseDType.toAlgKernel) c.grid (A.flattenState input) final) ∧
      (∀ idx i,active c i → final.readMem (Flat.address c A idx i).1 (Flat.address c A idx i).2=
        Whole.result c input idx i) ∧
      BlockState.WriteWithin (Flat.writes c A) (A.flattenState input) final := by
  have hs := Contract.safety admission
  exact AttentionFwdTriton3ScalarCast.RawKernel.whole_real c hs.valid A hs.allocated hs.disjoint closed input defined

/-- Standard public Real entry. Actual shared Tensor input/output address
transport is proved in `TensorBridge.complete_real_tensors`; the kernel theorem
itself takes the ideal Real state and proves successful complete execution. -/
theorem correct (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (args : Checked.Arguments) (admission : Contract.Admission c A p args)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (input : BlockState) (defined : input.undef=(fun _ _=>0)) :
    ∃ final, Nonempty (Kernel.GridLaunchedOrdinary
      (A.flattenKernel (Kernels.source c).eraseDType.toAlgKernel) c.grid (A.flattenState input) final) ∧
      (∀ idx lane,active c lane → final.readMem (Flat.address c A idx lane).1 (Flat.address c A idx lane).2=
        Whole.result c input idx lane) ∧
      BlockState.WriteWithin (Flat.writes c A) (A.flattenState input) final :=
  whole_real c A p args admission closed input defined

open TensorBinding in
theorem complete_real_tensors (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (a : Admission c A p r)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0) (heap : Nat → ℝ) :
    ∃ final,Nonempty (Kernel.GridLaunchedOrdinary
      (A.flattenKernel (Kernels.source c).eraseDType.toAlgKernel) c.grid
      (A.flattenState (initialState A heap)) final) ∧
      (∀ idx lane,active c lane →
        final.readMem (Flat.address c A idx lane).1 (Flat.address c A idx lane).2=
          Whole.result c (initialState A heap) idx lane) ∧
      (∀ z h row dim,(hz : z<c.Z) → (hh : h<c.H) → (hr : row<c.NQ) → (hd : dim<c.D) →
        ∃ idx lane,active c lane ∧
          Ownership.address c idx lane=(region .output,c.o.at z h row dim) ∧
          completedValue A final (r .output).tensor (index4 c.H c.rounded c.D z h row dim)=
            Whole.result c (initialState A heap) idx lane) ∧
      BlockState.WriteWithin (Flat.writes c A) (A.flattenState (initialState A heap)) final := by
  obtain ⟨final,execution,values,frame⟩ := whole_real c A p (arguments r) a.checked closed (initialState A heap) rfl
  refine ⟨final,execution,values,?_,frame⟩
  intro z h row dim hz hh hr hd
  have index := output_index c A p r a z h row dim hz hh (query_within_workspace c row hr) hd
  obtain ⟨idx,lane,live,coordinate⟩ := Return.output_covered c (Contract.safety a.checked).valid
    z h row dim hz hh hr hd
  refine ⟨idx,lane,live,coordinate,?_⟩
  unfold completedValue logicalValue
  rw [admitted_resolved c A p r a .output]
  simp only [Bool.false_eq_true,if_false]
  rw [allocation_address c A p r a .output _ index.1,index.2,Nat.mul_div_cancel_left _ (by decide : 0<4)]
  simpa only [Flat.address,coordinate] using values idx lane live

#axiomsClean whole_real
#axiomsClean correct
#axiomsClean complete_real_tensors
end Mathematical

section Native
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership TensorBinding Native
open VeriTile.Bench.Repairs.LightningAttention.KernelOnly (load32)

/-- The public native certificate constructs complete quotation execution.
Only same-port scalar/fixed-K8 instruction laws and an explicit structural
schedule are numerical interpretation boundaries. The six actual Tensors,
scale ABI word, all padded stores and exterior bytes use one shared memory.
Neither successful execution nor a kernel-output equality is a premise. -/
theorem fp_equiv (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (admission : Admission c A p r)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (schedule : Raw.Schedule c) (actual reference : Raw.Ops) (laws : Raw.LocalLaws actual reference)
    (before : Bytes.Memory) (scale : Native.Word) :
    let shape := shape_of_admission c A p r admission
    ∃ final,Reference.Composition.schedule backend.algebra
      (Reference.launched (ScaleBinding.template c)) c.grid (Reference.Composition.allPrograms c.grid)
      (Native.initialState A before)=some final ∧
      Bytes.publish c A (wordMemory c shape schedule actual scale final.mem) before=
        targetBytes c A shape schedule reference before scale ∧
      SupportedMemory (ScaleBinding.template c) final ∧ TaggedMemory final ∧
      (∀ idx lane,active c lane →
        Torch.FP32.logicalValue
          (fun addr => ⟨load32 (Bytes.publish c A (wordMemory c shape schedule actual scale final.mem) before) addr⟩)
          (r (laneRole lane)).tensor (admission.laws (laneRole lane)).tensor (laneIndex c idx lane)=
          ⟨projectLane c shape schedule reference scale (targetState c A before).mem idx lane⟩) ∧
      (∀ addr,(¬∃ idx lane byte,active c lane ∧ Physical.address c A idx lane byte=addr) →
        Bytes.publish c A (wordMemory c shape schedule actual scale final.mem) before addr=before addr) := by
  dsimp only
  let shape := shape_of_admission c A p r admission
  obtain ⟨final,execution,publication,supported,tagged,words,frame⟩ :=
    AttentionFwdTriton3ScalarCast.RawKernel.fp_equiv c A (Contract.safety admission.checked).valid shape
      schedule actual reference laws before scale
  refine ⟨final,execution,publication,supported,tagged,?_,frame⟩
  · intro idx lane live
    rw [publication]
    let words := wordMemory c shape schedule reference scale (targetState c A before).mem
    have readback := padded_tensor_readback c A p r admission closed words before idx lane live
    have projected := wordMemory_readback c (Contract.safety admission.checked).valid shape
      schedule reference scale (targetState c A before).mem idx lane live
    exact readback.trans (congrArg (fun word => (⟨word⟩ : Torch.FP32.Word)) projected)
/-- Every actual returned Tensor element is read from the independent native
target after the ordinary wrapper copies. M/L identities and unfinalized O
identity come from the storage-only API contract. -/
theorem returned_tensors (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (admission : Admission c A p r)
    (schedule : Raw.Schedule c) (reference : Raw.Ops) (before : Bytes.Memory) (scale : Native.Word)
    (after : Nat → Torch.FP32.Word) (returned : TensorReturn.Tuple)
    (api : TensorReturn.API c r
      (fun addr => ⟨load32 (targetBytes c A (shape_of_admission c A p r admission) schedule reference before scale) addr⟩)
      after returned) :
    returned.center=(r .center).tensor ∧ returned.denominator=(r .denominator).tensor ∧
    (c.finish=0 → returned.output=(r .output).tensor) ∧
    (∀ i,i<returned.output.metadata.numel →
      Torch.FP32.logicalValue after returned.output api.outputCopy.destinationLaws i=
        Torch.FP32.logicalValue
          (fun addr => ⟨load32 (targetBytes c A (shape_of_admission c A p r admission) schedule reference before scale) addr⟩)
          (r .output).tensor api.outputCopy.sourceLaws (TensorReturn.outputMap c i)) ∧
    (∀ i,i<returned.center.metadata.numel →
      Torch.FP32.logicalValue after returned.center api.centerCopy.destinationLaws i=
        Torch.FP32.logicalValue
          (fun addr => ⟨load32 (targetBytes c A (shape_of_admission c A p r admission) schedule reference before scale) addr⟩)
          (r .center).tensor api.centerCopy.sourceLaws i) ∧
    (∀ i,i<returned.denominator.metadata.numel →
      Torch.FP32.logicalValue after returned.denominator api.denominatorCopy.destinationLaws i=
        Torch.FP32.logicalValue
          (fun addr => ⟨load32 (targetBytes c A (shape_of_admission c A p r admission) schedule reference before scale) addr⟩)
          (r .denominator).tensor api.denominatorCopy.sourceLaws i) := by
  exact ⟨api.centerIdentity,api.denominatorIdentity,api.outputIdentity,
    TensorReturn.returned_values c r _ after returned api⟩

#axiomsClean fp_equiv
#axiomsClean returned_tensors
end Native
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.TensorBridge
