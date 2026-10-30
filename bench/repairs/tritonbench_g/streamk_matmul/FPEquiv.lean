import bench.repairs.tritonbench_g.streamk_matmul.Kernels
import bench.repairs.tritonbench_g.streamk_matmul.Proofs.Index

namespace VeriTile.Bench.Repairs.StreamK.Checked.FPEquiv
open Contract TensorBinding Whole001 ByteMemory
noncomputable section
set_option maxHeartbeats 4000000

/-- Whole checked native result with only fixed MMA8, scalar half conversion,
zero and atomic FP32-add correspondences. The ordered stream recurrence and
full-tile FP16 round trip remain distinct. -/
theorem matmul_checked_if_local (actual : Native.Instructions) (reference : Native.Reference)
    (numeric : Native.Numerical actual reference)
    (allocate : Host.Allocator Word) (dispatch : Host.Dispatch Word) (before : Read) (a b : Tensor) (la : Laws a) (lb : Laws b)
    (config : Config) (checked : checkTensorInputs a.metadata b.metadata config=.ok ())
    (p : Host.Prepared Word)
    (prepared : Host.prepare allocate a b config (fun address => readWord before address)=some p)
    (bytesFor : Host.Prepared Word → Read)
    (allocation : ByteHost.Allocation a b config before (bytesFor p) p)
    (ready : Host.Ready dispatch a b config p)
    (schedules : ∀ profile : Profile config,
      Schedule.Order (Source.firstWave (parameters a b config profile) (Native.operations actual)).length ×
      Schedule.CTAInterleaving (parameters a b config profile) ×
      Schedule.Order (Source.fullWave (parameters a b config profile) (Native.operations actual)).length) :
    let h := (checkTensorInputs_iff a.metadata b.metadata config).mp checked
    let params := parameters a b config h.profile
    let target := Returned.value params (Native.operations (Native.referenceInstructions reference))
      (left (fun address => readWord before address) a b config h.profile)
      (right (fun address => readWord before address) a b config h.profile)
    ∃ final,
      ByteHost.call (Native.operations actual) allocate dispatch a b config before bytesFor schedules=some (p.output,final) ∧
      p.output.metadata.shape=[rows a.metadata,columns b.metadata] ∧
      p.output.metadata.dtype=.floating .fp32 ∧ p.output.metadata.device=a.metadata.device ∧
      (∀ cell : Returned.Cell params,
        read32 final (Host.base p.output+4*(Returned.index params cell).val)=(target cell).bits) ∧
      (∀ cell : Returned.Cell params,∀ lane,lane<4 →
        final (Host.base p.output+4*(Returned.index params cell).val+lane)=byte (target cell).bits lane) ∧
      (∀ cell : Returned.Cell params,∀ lane,lane<4 →
        Host.base p.output+4*(Returned.index params cell).val+lane<2^64) ∧
      (∀ address,(a.storageBase≤address ∧ address<a.storageBase+4*a.storageElements) ∨
        (b.storageBase≤address ∧ address<b.storageBase+4*b.storageElements) → final address=before address) ∧
      (∀ address,Host.Outside p.output address → Host.OutsideScratch p address → final address=before address) ∧
      (¬Empty a.metadata b.metadata → ABI.Valid params ∧
        Index.Conformance a b config h.profile (Native.operations actual) p.read) := by
  dsimp only
  let h := (checkTensorInputs_iff a.metadata b.metadata config).mp checked
  let params := parameters a b config h.profile
  let ops := Native.operations actual
  let schedule := schedules h.profile
  let logical := (Host.returned ops a b config h.profile p schedule.1 schedule.2.1 schedule.2.2).2
  let final := ByteExecution.rendered (bytesFor p) (ByteHost.layout a b config h.profile p) logical
  have inputs := Host.prepared_inputs ops.zero (fun address => readWord before address) a b config la lb h p allocation.tensors
  have values : ∀ cell : Returned.Cell params,logical.output (Returned.index params cell)=
      Returned.value params (Native.operations (Native.referenceInstructions reference))
        (left (fun address => readWord before address) a b config h.profile)
        (right (fun address => readWord before address) a b config h.profile) cell := by
    intro cell
    change (Schedule.call _ ops _ _ _ _).output _=_
    rw [Schedule.call_eq]
    have result := Native.returned_fp params actual reference numeric
      (Host.initial ops.zero a b config h.profile p) cell
    change (Source.call _ ops _).output _=_ at result
    rw [inputs.1,inputs.2] at result
    exact result
  refine ⟨final,?_,allocation.tensors.outputShape,allocation.tensors.outputLaws.dtype,
    allocation.tensors.outputDevice,?_,?_,?_,?_,?_,fun nonempty =>
      ⟨Envelope.valid a b config h nonempty,Index.conformance a b config h nonempty ops p.read⟩⟩
  · rw [ByteHost.call_prepared ops allocate dispatch a b config before bytesFor h p prepared ready schedules,
      ByteHost.result_render ops a b config h.profile before (bytesFor p) p allocation]
  · intro cell
    change read32 (render _ (ByteExecution.cView (ByteHost.layout a b config h.profile p)) (extend logical.output))
      ((ByteExecution.cView (ByteHost.layout a b config h.profile p)).base+4*(Returned.index params cell).val)=_
    rw [render_read _ (ByteExecution.cView (ByteHost.layout a b config h.profile p)) _
      (Returned.index params cell).val (Returned.index params cell).isLt,extend_at,values cell]
  · intro cell lane hl
    change render _ (ByteExecution.cView (ByteHost.layout a b config h.profile p)) (extend logical.output)
      ((ByteExecution.cView (ByteHost.layout a b config h.profile p)).base+4*(Returned.index params cell).val+lane)=_
    rw [render_at _ (ByteExecution.cView (ByteHost.layout a b config h.profile p)) _
      (Returned.index params cell).val lane (Returned.index params cell).isLt hl,extend_at,values cell]
  · intro cell lane hl
    change Host.base p.output+4*(Returned.index params cell).val+lane<2^64
    have hi := (Returned.index params cell).isLt
    have window := allocation.tensors.outputWindow
    rw [Host.output_numel a b config _ p allocation.tensors] at window
    change (Returned.index params cell).val<rows a.metadata*columns b.metadata at hi
    omega
  · exact ByteHost.input_frame a b config h.profile before (bytesFor p) p allocation logical
  · exact ByteHost.rendered_frame a b config h.profile before (bytesFor p) p allocation logical

#axiomsClean matmul_checked_if_local
#specNonCircular Native.Numerical avoiding [Source.call,Schedule.call,Memory.run,Returned.value]
end
end VeriTile.Bench.Repairs.StreamK.Checked.FPEquiv
