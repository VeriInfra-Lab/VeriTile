import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Binding

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Correct
open VeriTile.Triton FP.Structural
noncomputable section

theorem python_checks_iff_preconditions (x : VeriTile.Torch.ArgumentMetadata) (pos : Contract.Position) :
    Contract.checkArguments x pos=.ok () ↔ Contract.AdmittedArguments x pos :=
  Contract.checkArguments_iff x pos

/-- Complete signed-position logical rotation and returned allocation in the
ideal Real interpretation. No floating-point accuracy is claimed here. -/
theorem whole_real (api : Memory.API ℝ Store Byte) (r : Request) (input output : RegionName)
    (s : State ℝ) (call : Native.Preparation api id r input s)
    (_abi : Native.IndexABI api id r input s call) :
    (Native.entry api id id ideal r input output s call).tensor=call.output ∧
    (Native.entry api id id ideal r input output s call).tensor.shape=[r.batches,r.rows,r.columns] ∧
    (∀ b m c, b<r.batches → m<r.rows → c<r.columns →
      Native.logicalRead api (Native.entry api id id ideal r input output s call).memory
        (Native.entry api id id ideal r input output s call).tensor b m c =
          Spec.result r (fun o => (s.mem input o).read .fp32) b m c) ∧
    (∀ b, call.caller b → api.byte (Native.entry api id id ideal r input output s call).memory b=api.byte call.before b) ∧
    (∀ l ∈ schedule r, ∃ final,
      FP.Structural.exec ideal.backend (kernel r input output) (programState s l)=some final ∧
      (active r l → (final.mem output (r.offset (batch l) (row r l) (column r l))).read .fp32 =
        Spec.result r (fun o => (s.mem input o).read .fp32) (batch l) (row r l) (column r l))) := by
  obtain ⟨hd,hs,hv,hb,hf⟩ := Native.whole api id id ideal r input output s call
  refine ⟨hd,hs,?_,hf,?_⟩
  · intro b m c hB hM hC
    rw [hv b m c hB hM hC]
    exact ideal_result r _ b m c
  · intro l hl
    obtain ⟨final,he,hv⟩ := source_execution ideal r input output s l
    refine ⟨final,he,?_⟩
    intro ha
    rw [hv ha,ideal_result]

#axiomsClean whole_real

theorem rbe_triton_transform_repaired_real_returns
    (api : Memory.API ℝ Store Byte) (r : Request) (input output : RegionName)
    (s : State ℝ) (call : Native.Preparation api id r input s) (binding : Native.Binding call) :
    (Native.entry api id id ideal r input output s call).tensor=call.output ∧
    call.output.shape=call.input.shape ∧ call.output.device=call.input.device ∧
    call.output.dtype="float32" ∧ call.output.elementBytes=4 ∧
    (∀ b m c,b<r.batches → m<r.rows → c<r.columns →
      Native.logicalRead api (Native.entry api id id ideal r input output s call).memory call.output b m c=
        Spec.result r (fun o => api.read call.before (call.input.base+4*o)) b m c) ∧
    (∀ offset,offset<call.input.storageBytes →
      api.byte (Native.entry api id id ideal r input output s call).memory (call.input.storageBase+offset)=
        api.byte call.before (call.input.storageBase+offset)) := by
  obtain ⟨hd,_,hv,_,_⟩ := whole_real api r input output s call (Native.checked_index_abi call binding)
  refine ⟨hd,call.outputShape.trans call.inputShape.symm,call.sameDevice,
    call.outputType.1,call.outputType.2,?_,Native.input_backing_frame api id id ideal r input output s call⟩
  intro b m c hb hm hc
  exact (hv b m c hb hm hc).trans (Native.original_input_spec api r input s call b m c hb hm hc)

#axiomsClean rbe_triton_transform_repaired_real_returns
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.Correct
