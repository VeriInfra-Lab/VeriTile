import bench.tritonbench_g.rms_norm_triton.Proofs.WholeHost

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

structure Runtime (α : Type) (input weight : Tensor) (normalized : List Nat)
    (entry : Memory α) (preexisting : Nat → Prop) where
  copyInput : Except String (Copied α input entry preexisting)
  copyWeight : (xc : Copied α input entry preexisting) →
    Except String (Copied α weight xc.memory (fun a => preexisting a ∨ a = xc.tensor.allocation))
  allocate : (xc : Copied α input entry preexisting) →
    (wc : Copied α weight xc.memory (fun a => preexisting a ∨ a = xc.tensor.allocation)) →
    Except String (Allocated α xc.tensor wc.memory
      (fun a => preexisting a ∨ a = xc.tensor.allocation ∨ a = wc.tensor.allocation))
  setup : Prepared α input weight entry preexisting → Except String (Schedule input normalized)

inductive Outcome (α : Type) where
  | inputCopyFailure (reason : String)
  | weightCopyFailure (reason : String) (memory : Memory α)
  | allocationFailure (reason : String) (memory : Memory α)
  | runtimeFailure (reason : String) (output : Tensor) (memory : Memory α)
  | kernelFailure
  | returned (output : Tensor) (memory : Memory α)

/-- Actual source order: metadata, x.contiguous, weight.contiguous, empty_like,
device/launch setup, complete original grid and returned allocated tensor.
Runtime callbacks supply only ordinary copy/allocation facts and a schedule. -/
def executeHost [Inhabited α] (mdl : Algebra α) (eps : ℝ)
    (rt : Runtime α input weight normalized entry preexisting) : Outcome α :=
  match rt.copyInput with
  | .error reason => .inputCopyFailure reason
  | .ok xc =>
    match rt.copyWeight xc with
    | .error reason => .weightCopyFailure reason xc.memory
    | .ok wc =>
      match rt.allocate xc wc with
      | .error reason => .allocationFailure reason wc.memory
      | .ok output =>
        let a : Prepared α input weight entry preexisting := ⟨xc,wc,output⟩
        match rt.setup a with
        | .error reason => .runtimeFailure reason output.tensor output.memory
        | .ok schedule =>
          match a.entryRun mdl normalized eps schedule with
          | none => .kernelFailure
          | some final => .returned output.tensor (a.commit final)

def ReturnedSpec [Inhabited α] (formula : Nat → Nat → α)
    (a : Prepared α input weight entry preexisting) (normalized : List Nat) (memory : Memory α) : Prop :=
  a.output.tensor.shape = input.shape ∧ a.output.tensor.dtype = input.dtype ∧
  a.output.tensor.device = input.device ∧ Contiguous a.output.tensor ∧
  (∀ row : Fin (rows input normalized), ∀ i : Fin (columns normalized),
    a.output.tensor.read memory (row.val*columns normalized+i.val) = formula row.val i.val) ∧
  (∀ allocation offset, PhysicalOutside a allocation offset →
    memory allocation offset = a.output.memory allocation offset) ∧
  (∀ allocation offset, preexisting allocation → memory allocation offset = entry allocation offset) ∧
  (∀ i, input.read memory i = input.read entry i) ∧
  (∀ i, weight.read memory i = weight.read entry i)

def HostSpec [Inhabited α] (formula : Nat → Nat → α)
    (rt : Runtime α input weight normalized entry preexisting) (outcome : Outcome α) : Prop :=
  (∃ reason, rt.copyInput = .error reason ∧ outcome = .inputCopyFailure reason) ∨
  (∃ xc reason, rt.copyInput = .ok xc ∧ rt.copyWeight xc = .error reason ∧
    outcome = .weightCopyFailure reason xc.memory ∧
    ∀ allocation offset, preexisting allocation → xc.memory allocation offset = entry allocation offset) ∨
  (∃ xc wc reason, rt.copyInput = .ok xc ∧ rt.copyWeight xc = .ok wc ∧
    rt.allocate xc wc = .error reason ∧ outcome = .allocationFailure reason wc.memory ∧
    ∀ allocation offset, preexisting allocation → wc.memory allocation offset = entry allocation offset) ∨
  (∃ xc wc output reason, rt.copyInput = .ok xc ∧ rt.copyWeight xc = .ok wc ∧
    rt.allocate xc wc = .ok output ∧ rt.setup ⟨xc,wc,output⟩ = .error reason ∧
    outcome = .runtimeFailure reason output.tensor output.memory ∧
    ∀ allocation offset, preexisting allocation → output.memory allocation offset = entry allocation offset) ∨
  (∃ xc wc output schedule final, rt.copyInput = .ok xc ∧ rt.copyWeight xc = .ok wc ∧
    rt.allocate xc wc = .ok output ∧ rt.setup ⟨xc,wc,output⟩ = .ok schedule ∧
    outcome = .returned output.tensor ((Prepared.mk xc wc output).commit final) ∧
    ReturnedSpec formula ⟨xc,wc,output⟩ normalized ((Prepared.mk xc wc output).commit final))

def sourceFormula [Inhabited α] (mdl : Algebra α) (input : Tensor) (normalized : List Nat)
    (eps : ℝ) (entryInput entryWeight : Nat → α) (row i : Nat) : α :=
  rowValue mdl ⟨"X","W","Y",rows input normalized,columns normalized,
    nextPower (columns normalized),nextPower_covers _⟩ eps
    (fun j => entryInput (row*columns normalized+j)) entryWeight i

theorem host_outcomes [Inhabited α] (mdl : Algebra α) (eps : ℝ)
    (rt : Runtime α input weight normalized entry preexisting)
    (domain : InputDomain input weight normalized preexisting) :
    HostSpec (sourceFormula mdl input normalized eps (input.read entry) (weight.read entry)) rt
      (executeHost mdl eps rt) := by
  cases hx : rt.copyInput with
  | error reason => exact Or.inl ⟨reason,hx,by simp [executeHost,hx]⟩
  | ok xc =>
    cases hw : rt.copyWeight xc with
    | error reason =>
      exact Or.inr (Or.inl ⟨xc,reason,hx,hw,by simp [executeHost,hx,hw],xc.frame⟩)
    | ok wc =>
      cases ha : rt.allocate xc wc with
      | error reason =>
        refine Or.inr (Or.inr (Or.inl ⟨xc,wc,reason,hx,hw,ha,by simp [executeHost,hx,hw,ha],?_⟩))
        intro allocation offset existed
        exact (wc.frame _ _ (Or.inl existed)).trans (xc.frame _ _ existed)
      | ok output =>
        let a : Prepared α input weight entry preexisting := ⟨xc,wc,output⟩
        cases hs : rt.setup a with
        | error reason =>
          refine Or.inr (Or.inr (Or.inr (Or.inl
            ⟨xc,wc,output,reason,hx,hw,ha,hs,?_,?_⟩)))
          · simp [executeHost,hx,hw,ha,hs,a] at *
          · exact a.entry_frame
        | ok schedule =>
          obtain ⟨final,ran,values,frame,entryFrame⟩ := a.entry_result mdl normalized eps domain schedule
          refine Or.inr (Or.inr (Or.inr (Or.inr
            ⟨xc,wc,output,schedule,final,hx,hw,ha,hs,?_,?_⟩)))
          · simpa [executeHost,hx,hw,ha,hs,a] using congrArg
              (fun value : Option (State α) =>
                match value with
                | none => Outcome.kernelFailure
                | some result => Outcome.returned output.tensor (a.commit result)) ran
          · refine ⟨output.shape.trans xc.shape,output.dtype.trans xc.dtype,
              output.device.trans xc.device,output.contiguous,values,frame,entryFrame,?_,?_⟩
            · intro i
              exact entryFrame input.allocation (input.offset i) domain.inputExisted
            · intro i
              exact entryFrame weight.allocation (weight.offset i) domain.weightExisted

theorem real_host_outcomes (eps : ℝ)
    (rt : Runtime ℝ input weight normalized entry preexisting)
    (domain : InputDomain input weight normalized preexisting) :
    HostSpec (fun row i => realFormula (columns normalized) eps
      (fun j => input.read entry (row*columns normalized+j)) (weight.read entry) i) rt
      (executeHost realAlgebra eps rt) := by
  have formula : sourceFormula realAlgebra input normalized eps (input.read entry) (weight.read entry) =
      (fun row i => realFormula (columns normalized) eps
        (fun j => input.read entry (row*columns normalized+j)) (weight.read entry) i) := by
    funext row i
    exact rowValue_real _ _ _ _ _
  rw [← formula]
  exact host_outcomes realAlgebra eps rt domain

end
#guard_msgs (drop info) in
#specNonCircular sourceFormula avoiding [Layout.source,sourceRun,rowExec,runRows,Prepared.entryRun,executeHost]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Whole
