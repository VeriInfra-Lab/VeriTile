import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Proofs.PhysicalIO
import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Misc.RawNative
import bench.repairs.tritonbench_g.token_attn_mistral_grid_y_checked.Proofs.Indices

/-! Full original Tensor applications. The checked parameter, separation and
unique-store lemmas instantiate the same raw Grid.whole execution theorem used
by the primary entries; logical/physical readback and frames remain unchanged. -/
namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.TensorBridge
open VeriTile.Triton FP.Structural TokenAttnMistral
noncomputable section
set_option maxHeartbeats 4000000

/-- The actual quoted kernel succeeds for every permutation of the complete
submitted grid. Every caller logical output equals the independent gathered
weighted sum, and the full source and physical frames are preserved. -/
theorem kernel_correct (c : TensorCall) (laws : TensorLaws c) (metadata : Reader)
    (read : Nat → ℝ) (contract : InputContract c.metadata metadata)
    (work : List Grid.PID)
    (order : (Grid.fullGrid (batches c.metadata) (heads c.metadata)).Perm work) :
    ∃ after,Grid.run Real.algebra Real.sum (parameters c.metadata) work
        (view c.metadata read metadata (fun a => .mk .fp32 (read a)))=some after ∧
      (∀ b < batches c.metadata,∀ h < heads c.metadata,∀ d : Fin (width c.metadata),
        logical (observe id c.metadata read after) c .output (outputIndex c.metadata b h d.val)=
          logicalRealTarget c metadata read b h d.val) ∧
      RawFrame c.metadata (view c.metadata read metadata (fun a => .mk .fp32 (read a))) after ∧
      (∀ a,Exterior c.metadata a → observe id c.metadata read after a=read a) ∧
      (0 < batches c.metadata → ∀ r∈reads,∀ i < (c r).metadata.numel,
        logical (observe id c.metadata read after) c r i=logical read c r i) := by
  obtain ⟨after,run,values,frame⟩ := complete Real.algebra Real.sum c.metadata
    (view c.metadata read metadata (fun a => .mk .fp32 (read a))) work order
  refine ⟨after,run,?_,frame,?_,?_⟩
  · intro b hb h hh d
    have hi := output_bound c.metadata metadata contract b h d.val hb hh d.isLt
    rw [logical_read _ c laws contract.header _ _ hi,observe_store _ _ _ _ _ hi,
      values b hb h hh d]
    simpa only [Cell.read,id_eq] using
      (real_value c.metadata metadata read (fun a => .mk .fp32 (read a)) b h d).trans
        (real_logical c laws metadata read contract b h d.val hb hh d.isLt)
  · exact fun a ha => observe_frame _ _ _ _ a ha
  · intro live r hr i hi
    unfold logical
    apply observe_frame
    rw [tensor_address c laws contract.header r i hi]
    exact read_exterior c.metadata metadata contract live r hr i hi

/-- Empty calls submit no grid and leave every caller word unchanged. -/
theorem empty_call {α β : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (m : Inputs) (metadata : Reader) (contract : InputContract m metadata)
    (empty : batches m=0) (memory : Grid.Memory α) (work : List Grid.PID)
    (order : (Grid.fullGrid (batches m) (heads m)).Perm work)
    (decode : α → β) (before : Nat → β) :
    work=[] ∧ Grid.run M R (parameters m) work memory=some memory ∧
    ∀ a,observe decode m before memory a=before a := by
  have nil : work=[] := List.nil_perm.mp (by simpa [Grid.fullGrid,empty] using order)
  have size : (m .output).numel=0 := by
    simp [VeriTile.Torch.TensorMetadata.numel,contract.header.outputShape,empty]
  refine ⟨nil,by simp [nil,Grid.run],?_⟩
  intro a
  unfold observe
  rw [if_neg (by omega)]

#axiomsClean kernel_correct
#axiomsClean empty_call

set_option maxHeartbeats 4000000

/-- Native success and every logical output word. Instructions supplies only
add/mul/FMA primitives; R chooses complete ordered reduction trees and S only
legal adjacent contraction. There is no supplied row or kernel result. The
same caller word map supplies FP32 values and signed int32 metadata. -/
theorem kernel_native (I : Native.Instructions) (R : Native.Reductions) (S : Native.Schedule)
    (c : TensorCall) (laws : TensorLaws c) (words : Reader)
    (contract : InputContract c.metadata words) (work : List Grid.PID)
    (order : (Grid.fullGrid (batches c.metadata) (heads c.metadata)).Perm work) :
    ∃ after,Grid.run Native.algebra (Native.reduction R) (parameters c.metadata) work
        (Native.entry (view c.metadata words words (fun a => .mk .fp32 (words a))))=some after ∧
      (∀ b < batches c.metadata,∀ h < heads c.metadata,∀ d : Fin (width c.metadata),
        logical (observe (Native.eval I S) c.metadata words after) c .output
          (outputIndex c.metadata b h d.val)=NativeTarget.reference I R S c.metadata words words b h d) ∧
      RawFrame c.metadata (Native.entry (view c.metadata words words (fun a => .mk .fp32 (words a)))) after ∧
      (∀ a,Exterior c.metadata a → observe (Native.eval I S) c.metadata words after a=words a) ∧
      (0 < batches c.metadata → ∀ r∈reads,∀ i < (c r).metadata.numel,
        logical (observe (Native.eval I S) c.metadata words after) c r i=logical words c r i) := by
  obtain ⟨after,run,values,frame⟩ := complete Native.algebra (Native.reduction R) c.metadata
    (Native.entry (view c.metadata words words (fun a => .mk .fp32 (words a)))) work order
  refine ⟨after,run,?_,frame,?_,?_⟩
  · intro b hb h hh d
    have hi := output_bound c.metadata words contract b h d.val hb hh d.isLt
    rw [logical_read _ c laws contract.header _ _ hi,observe_store _ _ _ _ _ hi,
      values b hb h hh d]
    simp only [Cell.read,NativeTarget.reference]
    exact congrArg (Native.eval I S)
      (NativeTarget.term_eq R c.metadata words words (fun a => .mk .fp32 (words a)) b h d)
  · exact fun a ha => observe_frame _ _ _ _ a ha
  · intro live r hr i hi
    unfold logical
    apply observe_frame
    rw [tensor_address c laws contract.header r i hi]
    exact read_exterior c.metadata words contract live r hr i hi

#axiomsClean kernel_native

end
end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked.TensorBridge
