import bench.repairs.tritonbench_g.dropout_triton.Proofs.Wrapper
import bench.repairs.tritonbench_g.dropout_triton.Proofs.Pointers

/-! Composition of the admitted wrapper with the same allocated object's
physical storage. The interfaces below contain only Tensor/allocator laws. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.PhysicalWrapper
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.Finite
noncomputable section

def scalar {α : Type} (M : Algebra α) (p : ℝ) (x : α) (keep : Int) : α :=
  if keep ≠ 0 then M.binary (some .fp32) .real .div (M.fp32Load x)
    (M.binary (some .fp32) .real .sub
      (M.literal (some .fp32) .real 1) (M.literal (some .fp32) .real p))
  else M.literal (some .fp32) .real 0

/-- Ordinary contiguous Tensor reads at wrapper entry. Both views may refer
to overlapping storage. No post-execution value is assumed. -/
def EntryBinding {α S : Type} [Inhabited α] (ops : Physical.Memory S α Int)
    (x keep : TensorMetadata) (wx : Physical.TensorWindow x .fp32)
    (wk : Physical.TensorWindow keep .i32) (logical : State α) (physical : S) : Prop :=
  ∀ i, i < x.numel →
    Physical.readX ops physical wx.view i = (logical.mem "x" i).read .real ∧
    Physical.readK ops physical wk.view i = (logical.mem "keep" i).read .int

/-- Physical binding of the particular object identified by `a.outputObject`.
Its metadata and window describe that same successful empty_like allocation.
Allocation may initialize fresh output bytes arbitrarily; it preserves entry
input reads and every byte outside the returned tensor's active storage. -/
structure AllocationBinding {α S : Type} (ops : Physical.Memory S α Int)
    (x keep : TensorMetadata) (wx : Physical.TensorWindow x .fp32)
    (wk : Physical.TensorWindow keep .i32) (logical : State α) (entry : S)
    (a : Wrapper.Allocation x logical) where
  outputWindow : Physical.TensorWindow a.output .fp32
  state : S
  freshX : StorageDisjoint outputWindow.view wx.view
  freshKeep : StorageDisjoint outputWindow.view wk.view
  readX : ∀ i, i < x.numel → Physical.readX ops state wx.view i = Physical.readX ops entry wx.view i
  readKeep : ∀ i, i < x.numel → Physical.readK ops state wk.view i = Physical.readK ops entry wk.view i
  frame : ∀ address, ¬outputWindow.view.active x.numel address →
    ops.byte state address = ops.byte entry address

theorem native_output {α S : Type} (ops : Physical.Memory S α Int)
    (compute : α → Int → α) (x keep output : TensorMetadata)
    (h : TensorContract x keep)
    (wx : Physical.TensorWindow x .fp32) (wk : Physical.TensorWindow keep .i32)
    (wo : Physical.TensorWindow output .fp32) (shape : output.shape = x.shape)
    (freshX : StorageDisjoint wo.view wx.view) (freshKeep : StorageDisjoint wo.view wk.view)
    (before : S) :
    (∀ i, i < x.numel → Physical.readX ops
      (Physical.runNative ops compute wx.view wk.view wo.view x.numel before) wo.view i =
      compute (Physical.readX ops before wx.view i) (Physical.readK ops before wk.view i)) ∧
    (∀ address, ¬wo.view.active x.numel address → ops.byte
      (Physical.runNative ops compute wx.view wk.view wo.view x.numel before) address = ops.byte before address) := by
  by_cases empty : x.numel = 0
  · constructor
    · simp [empty]
    · intro address _
      simp [Physical.runNative, Physical.nativeBatches, gridCount, empty, Physical.runBatches]
  · have pos : 0 < x.numel := Nat.pos_of_ne_zero empty
    have keepCount := same_numel x keep h
    have outCount : output.numel = x.numel := congrArg List.prod shape
    exact Physical.complete_native ops compute x keep h wx.view wk.view wo.view before
      wx.type wo.type (wx.bounds pos)
      (by simpa only [←keepCount] using wk.bounds (keepCount ▸ pos))
      (by simpa only [outCount] using wo.bounds (outCount.symm ▸ pos)) freshX freshKeep

/-- Same successful guarded wrapper, allocated-object identity, metadata,
input-device launch and physical native execution. Output agreement and the
exterior byte frame are conclusions derived from the complete grid proofs. -/
theorem complete {α S : Type} [Inhabited α] (M : Algebra α)
    (ops : Physical.Memory S α Int) (allocate : Wrapper.Allocator α)
    (x keep : TensorMetadata) (p : ℝ) (h : InputContract x keep (.float (some p)))
    (wx : Physical.TensorWindow x .fp32) (wk : Physical.TensorWindow keep .i32)
    (before : State α) (entry : S) (inputs : EntryBinding ops x keep wx wk before entry)
    (a : AllocationBinding ops x keep wx wk before entry (allocate x before)) :
    ∃ final, Wrapper.run M allocate x keep (.float (some p)) before = .ok (some final) ∧
      final.outputObject = (allocate x before).outputObject ∧
      final.output = (allocate x before).output ∧
      final.launchDevice = (if x.numel = 0 then none else some x.device) ∧
      (0 < x.numel → a.outputWindow.view.address 0 = final.output.dataPtr.toNat) ∧
      a.outputWindow.view.device = x.device.index ∧
      (∀ i, i < x.numel →
        Physical.readX ops (Physical.runNative ops (scalar M p) wx.view wk.view
          a.outputWindow.view x.numel a.state) a.outputWindow.view i =
            (final.state.mem "out" i).read .real ∧
        (final.state.mem "out" i).read .real =
          scalar M p (Physical.readX ops entry wx.view i) (Physical.readK ops entry wk.view i)) ∧
      (∀ address, ¬a.outputWindow.view.active x.numel address → ops.byte
        (Physical.runNative ops (scalar M p) wx.view wk.view a.outputWindow.view x.numel a.state)
          address = ops.byte entry address) ∧
      (∀ (r : RegionName) i, r ≠ "out" ∨ x.numel ≤ i → final.state.mem r i = before.mem r i) := by
  obtain ⟨q, final, hp, returned, metadata, _, _, _, values, frame⟩ :=
    Wrapper.correct M allocate x keep (.float (some p)) h before
  have hq : p = q := by cases hp; rfl
  subst q
  have details := Wrapper.return_details M allocate x keep (.float (some p)) h before final returned
  have physical := native_output ops (scalar M p) x keep (allocate x before).output h.tensors
    wx wk a.outputWindow (allocate x before).shape a.freshX a.freshKeep a.state
  refine ⟨final, returned, details.1, metadata, details.2, ?_, ?_, ?_, ?_, frame⟩
  · intro positive
    rw [metadata]
    apply a.outputWindow.pointer
    simpa only [VeriTile.Torch.TensorMetadata.numel, (allocate x before).shape] using positive
  · exact a.outputWindow.device.trans (congrArg VeriTile.Torch.Device.index (allocate x before).device)
  · intro i hi
    have hv : (final.state.mem "out" i).read .real =
        scalar M p (Physical.readX ops entry wx.view i) (Physical.readK ops entry wk.view i) := by
      rw [values i hi]
      change scalar M p ((before.mem "x" i).read .real) ((before.mem "keep" i).read .int) = _
      rw [(inputs i hi).1, (inputs i hi).2]
    refine ⟨?_, hv⟩
    rw [physical.1 i hi, a.readX i hi, a.readKeep i hi, hv]
  · intro address outside
    exact (physical.2 address outside).trans (a.frame address outside)

#axiomsClean native_output
#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DropoutTriton.PhysicalWrapper
