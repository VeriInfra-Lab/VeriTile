import bench.repairs.tritonbench_g.streamk_matmul.Kernels
import bench.repairs.tritonbench_g.streamk_matmul.Proofs.ValueHost

namespace VeriTile.Bench.Repairs.StreamK.Checked.Correct
open Contract TensorBinding Whole001
noncomputable section

/-- Independent full-K matrix product over the original Tensor logical views.
The physical element-start reader is an arbitrary Real function. -/
def matrixProduct (read : ValueRead ℝ) (a b : Tensor) (i j : Nat) : ℝ :=
  ∑ k : Fin (columns a.metadata),logicalValue read a (i*columns a.metadata+k.val)*
    logicalValue read b (k.val*columns b.metadata+j)

theorem checks_iff (a b : VeriTile.Torch.ArgumentMetadata) (config : Config) :
    checkInputs a b config=.ok () ↔ AdmittedArguments a b config := checkInputs_iff a b config

/-- Checked wrapper, actual sequential allocator calls, finite launch executor,
returned C identity, every matrix entry, immutable inputs and exterior frame. -/
theorem matmul_checked_real_state (allocate : Host.Allocator ℝ) (dispatch : Host.Dispatch ℝ) (read : ValueRead ℝ)
    (a b : Tensor) (la : Laws a) (lb : Laws b) (config : Config)
    (checked : checkTensorInputs a.metadata b.metadata config=.ok ())
    (p : Host.Prepared ℝ) (prepared : Host.prepare allocate a b config read=some p)
    (allocation : Host.Allocation a b config read p)
    (ready : Host.Ready dispatch a b config p)
    (schedules : ∀ profile : Profile config,
      Schedule.Order (Source.firstWave (parameters a b config profile) Arithmetic.real).length ×
      Schedule.CTAInterleaving (parameters a b config profile) ×
      Schedule.Order (Source.fullWave (parameters a b config profile) Arithmetic.real).length) :
    let h := (checkTensorInputs_iff a.metadata b.metadata config).mp checked
    let schedule := schedules h.profile
    let final := (Host.returned Arithmetic.real a b config h.profile p schedule.1 schedule.2.1 schedule.2.2).2
    Host.call Arithmetic.real allocate dispatch a b config read schedules=some (p.output,⟨h.profile,final⟩) ∧
    p.output.metadata.shape=[rows a.metadata,columns b.metadata] ∧
    p.output.metadata.dtype=.floating .fp32 ∧ p.output.metadata.device=a.metadata.device ∧
    (∀ cell : Returned.Cell (parameters a b config h.profile),
      final.output (Returned.index _ cell)=matrixProduct read a b cell.1.val cell.2.val) ∧
    final.inputA=left read a b config h.profile ∧ final.inputB=right read a b config h.profile ∧
    final.exterior=(Host.initial 0 a b config h.profile p).exterior ∧
    (¬Empty a.metadata b.metadata → ABI.Valid (parameters a b config h.profile) ∧
      Index.Conformance a b config h.profile Arithmetic.real p.read) := by
  dsimp only
  let h := (checkTensorInputs_iff a.metadata b.metadata config).mp checked
  have input := Host.prepared_inputs (0 : ℝ) read a b config la lb h p allocation
  refine ⟨Host.call_prepared Arithmetic.real allocate dispatch a b config read h p prepared ready schedules,
    allocation.outputShape,allocation.outputLaws.dtype,allocation.outputDevice,?_,?_,?_,?_,
    fun nonempty => ⟨Envelope.valid a b config h nonempty,
      Index.conformance a b config h nonempty Arithmetic.real p.read⟩⟩
  · intro cell
    change (Schedule.call _ Arithmetic.real _ _ _ _).output _=_
    rw [Schedule.call_eq]
    have result := Returned.real_contract (parameters a b config h.profile)
      (Host.initial 0 a b config h.profile p) cell
    change (Source.call _ Arithmetic.real _).output _=_ at result
    rw [input.1,input.2] at result
    simpa only [Reference.matrixProduct,left,right,parameters,plan,Nat.mul_one,matrixProduct] using result
  · change (Schedule.call _ Arithmetic.real _ _ _ _).inputA=_
    rw [Schedule.call_eq]
    exact (Source.call_frame _ Arithmetic.real _).1.trans input.1
  · change (Schedule.call _ Arithmetic.real _ _ _ _).inputB=_
    rw [Schedule.call_eq]
    exact (Source.call_frame _ Arithmetic.real _).2.1.trans input.2
  · change (Schedule.call _ Arithmetic.real _ _ _ _).exterior=_
    rw [Schedule.call_eq]
    exact (Source.call_frame _ Arithmetic.real _).2.2

/-- Full checked ideal call on an unrestricted Real physical reader, including
actual output allocation identity, physical stores, input storage and frame. -/
theorem matmul_checked_real (allocate : Host.Allocator ℝ) (dispatch : Host.Dispatch ℝ) (before : ValueRead ℝ)
    (a b : Tensor) (la : Laws a) (lb : Laws b) (config : Config)
    (checked : checkTensorInputs a.metadata b.metadata config=.ok ())
    (p : Host.Prepared ℝ) (prepared : Host.prepare allocate a b config before=some p)
    (allocation : Host.Allocation a b config before p) (ready : Host.Ready dispatch a b config p)
    (schedules : ∀ profile : Profile config,
      Schedule.Order (Source.firstWave (parameters a b config profile) Arithmetic.real).length ×
      Schedule.CTAInterleaving (parameters a b config profile) ×
      Schedule.Order (Source.fullWave (parameters a b config profile) Arithmetic.real).length) :
    let h := (checkTensorInputs_iff a.metadata b.metadata config).mp checked
    let params := parameters a b config h.profile
    ∃ final,
      ValueHost.call Arithmetic.real allocate dispatch a b config before schedules=some (p.output,final) ∧
      p.output.metadata.shape=[rows a.metadata,columns b.metadata] ∧
      p.output.metadata.dtype=.floating .fp32 ∧ p.output.metadata.device=a.metadata.device ∧
      (∀ cell : Returned.Cell params,
        final (Host.base p.output+4*(Returned.index params cell).val)=matrixProduct before a b cell.1.val cell.2.val) ∧
      (∀ address,(a.storageBase≤address ∧ address<a.storageBase+4*a.storageElements) ∨
        (b.storageBase≤address ∧ address<b.storageBase+4*b.storageElements) → final address=before address) ∧
      (∀ address,Host.Outside p.output address → Host.OutsideScratch p address → final address=before address) ∧
      (¬Empty a.metadata b.metadata → ABI.Valid params ∧
        Index.Conformance a b config h.profile Arithmetic.real p.read) := by
  dsimp only
  let h := (checkTensorInputs_iff a.metadata b.metadata config).mp checked
  let params := parameters a b config h.profile
  let schedule := schedules h.profile
  let logical := (Host.returned Arithmetic.real a b config h.profile p schedule.1 schedule.2.1 schedule.2.2).2
  let final := ValueMemory.rendered p.read (ByteHost.layout a b config h.profile p) logical
  have inputs := Host.prepared_inputs (0 : ℝ) before a b config la lb h p allocation
  have values : ∀ cell : Returned.Cell params,logical.output (Returned.index params cell)=
      matrixProduct before a b cell.1.val cell.2.val := by
    intro cell
    change (Schedule.call _ Arithmetic.real _ _ _ _).output _=_
    rw [Schedule.call_eq]
    have result := Returned.real_contract params (Host.initial 0 a b config h.profile p) cell
    change (Source.call _ Arithmetic.real _).output _=_ at result
    rw [inputs.1,inputs.2] at result
    simpa only [Reference.matrixProduct,left,right,params,parameters,plan,Nat.mul_one,matrixProduct] using result
  refine ⟨final,?_,allocation.outputShape,allocation.outputLaws.dtype,allocation.outputDevice,?_,?_,?_,
    fun nonempty => ⟨Envelope.valid a b config h nonempty,
      Index.conformance a b config h nonempty Arithmetic.real p.read⟩⟩
  · rw [ValueHost.call_prepared Arithmetic.real allocate dispatch a b config before h p prepared ready schedules,
      ValueHost.result_render Arithmetic.real a b config h.profile before p allocation]
  · intro cell
    change ValueMemory.render _ (ByteHost.layout a b config h.profile p).outputBase logical.output
      ((ByteHost.layout a b config h.profile p).outputBase+4*(Returned.index params cell).val)=_
    rw [ValueMemory.render_at,values cell]
  · exact ValueHost.input_frame a b config h.profile before p allocation logical
  · exact ValueHost.rendered_frame a b config h.profile before p allocation logical

#axiomsClean matmul_checked_real
#axiomsClean matmul_checked_real_state
#specNonCircular matrixProduct avoiding [Source.call,Schedule.call,Memory.run,Returned.value,Reference.matrixProduct]
end
end VeriTile.Bench.Repairs.StreamK.Checked.Correct
