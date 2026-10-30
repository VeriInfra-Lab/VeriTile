import bench.tritonbench_g.rms_norm_triton.Proofs.WholeIndex
import bench.tritonbench_g.rms_norm_triton.Proofs.Observed

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed
open VeriTile.Triton FP.Structural
open Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

def projectOutcome (M : NativeModel α) (profile : ReductionFamily) (N : Nat) : Outcome (Term α) → Outcome α
  | .inputCopyFailure reason => .inputCopyFailure reason
  | .weightCopyFailure reason memory => .weightCopyFailure reason (projectMemory M profile N 0 memory)
  | .allocationFailure reason memory => .allocationFailure reason (projectMemory M profile N 0 memory)
  | .runtimeFailure reason output memory =>
      .runtimeFailure reason output (projectMemory M profile N output.base memory)
  | .kernelFailure => .kernelFailure
  | .returned output memory => .returned output (projectMemory M profile N output.base memory)

def executeHost (M : NativeModel α) (profile : ReductionFamily) (eps : ℝ)
    (rt : Runtime (Term α) input weight normalized (entryGraph entry) preexisting) : Outcome α :=
  projectOutcome M profile (columns normalized) (Whole.executeHost graphAlgebra eps rt)

def ReturnedSpec (M : NativeModel α) (profile : ReductionFamily)
    (a : Prepared (Term α) input weight (entryGraph entry) preexisting)
    (normalized : List Nat) (memory : Memory α) : Prop :=
  a.output.tensor.shape = input.shape ∧ a.output.tensor.dtype = input.dtype ∧
  a.output.tensor.device = input.device ∧ Contiguous a.output.tensor ∧
  (∀ row : Fin (rows input normalized), ∀ i : Fin (columns normalized),
    a.output.tensor.read memory (row.val*columns normalized+i.val) =
      reference M profile (columns normalized) (nextPower (columns normalized))
        (fun j => input.read entry (row.val*columns normalized+j)) (weight.read entry) i.val) ∧
  (∀ allocation offset, PhysicalOutside a allocation offset →
    memory allocation offset = projectMemory M profile (columns normalized) a.output.tensor.base a.output.memory allocation offset) ∧
  (∀ allocation offset, preexisting allocation → memory allocation offset = entry allocation offset) ∧
  (∀ i, input.read memory i = input.read entry i) ∧
  (∀ i, weight.read memory i = weight.read entry i)

def HostSpec (M : NativeModel α) (profile : ReductionFamily)
    (rt : Runtime (Term α) input weight normalized (entryGraph entry) preexisting) (outcome : Outcome α) : Prop :=
  (∃ reason, rt.copyInput = .error reason ∧ outcome = .inputCopyFailure reason) ∨
  (∃ xc reason, rt.copyInput = .ok xc ∧ rt.copyWeight xc = .error reason ∧
    outcome = .weightCopyFailure reason (projectMemory M profile (columns normalized) 0 xc.memory) ∧
    ∀ allocation offset, preexisting allocation →
      projectMemory M profile (columns normalized) 0 xc.memory allocation offset = entry allocation offset) ∨
  (∃ xc wc reason, rt.copyInput = .ok xc ∧ rt.copyWeight xc = .ok wc ∧
    rt.allocate xc wc = .error reason ∧ outcome = .allocationFailure reason (projectMemory M profile (columns normalized) 0 wc.memory) ∧
    ∀ allocation offset, preexisting allocation →
      projectMemory M profile (columns normalized) 0 wc.memory allocation offset = entry allocation offset) ∨
  (∃ xc wc output reason, rt.copyInput = .ok xc ∧ rt.copyWeight xc = .ok wc ∧
    rt.allocate xc wc = .ok output ∧ rt.setup ⟨xc,wc,output⟩ = .error reason ∧
    outcome = .runtimeFailure reason output.tensor (projectMemory M profile (columns normalized) output.tensor.base output.memory) ∧
    ∀ allocation offset, preexisting allocation →
      projectMemory M profile (columns normalized) output.tensor.base output.memory allocation offset = entry allocation offset) ∨
  (∃ xc wc output schedule final, rt.copyInput = .ok xc ∧ rt.copyWeight xc = .ok wc ∧
    rt.allocate xc wc = .ok output ∧ rt.setup ⟨xc,wc,output⟩ = .ok schedule ∧
    outcome = .returned output.tensor (projectMemory M profile (columns normalized) output.tensor.base ((Prepared.mk xc wc output).commit final)) ∧
    ReturnedSpec M profile ⟨xc,wc,output⟩ normalized
      (projectMemory M profile (columns normalized) output.tensor.base ((Prepared.mk xc wc output).commit final)))

theorem returned_projection (M : NativeModel α) (profile : ReductionFamily) (eps : ℝ)
    (a : Prepared (Term α) input weight (entryGraph entry) preexisting)
    (domain : InputDomain input weight normalized preexisting) (memory : Memory (Term α))
    (h : Whole.ReturnedSpec (sourceFormula graphAlgebra input normalized eps
      (input.read (entryGraph entry)) (weight.read (entryGraph entry))) a normalized memory) :
    ReturnedSpec M profile a normalized (projectMemory M profile (columns normalized) a.output.tensor.base memory) := by
  rcases h with ⟨shape,dtype,device,contiguous,values,frame,entryFrame,inputFrame,weightFrame⟩
  refine ⟨shape,dtype,device,contiguous,?_,?_,?_,?_,?_⟩
  · intro row i
    have shapeProduct := shape_product domain
    have bound : row.val*columns normalized+i.val < a.output.tensor.numel := by
      have hrow := row.isLt
      have hi := i.isLt
      rw [Tensor.numel,shape]
      change row.val*columns normalized+i.val < input.numel
      rw [shapeProduct]
      nlinarith
    have offset := contiguous _ bound
    have lane : ((a.output.tensor.offset (row.val*columns normalized+i.val)-a.output.tensor.base)%columns normalized) = i.val := by
      rw [offset]
      simpa [Nat.add_mod,Nat.mul_mod,Nat.mod_eq_of_lt i.isLt] using
        (show ((a.output.tensor.base+(row.val*columns normalized+i.val)-a.output.tensor.base)%columns normalized) = i.val by
          simp [Nat.add_mod,Nat.mul_mod,Nat.mod_eq_of_lt i.isLt])
    have value := congrArg (interpret M profile i.val) (values row i)
    simp only [sourceFormula] at value
    rw [show weight.read (entryGraph entry) = (fun j => Term.input (weight.read entry j)) from rfl] at value
    simpa only [Tensor.read,projectMemory,lane,sourceFormula,entryGraph,source_graph_lowering] using value
  · intro allocation offset outside
    exact congrArg (interpret M profile ((offset-a.output.tensor.base)%columns normalized)) (frame allocation offset outside)
  · intro allocation offset existed
    exact congrArg (interpret M profile ((offset-a.output.tensor.base)%columns normalized)) (entryFrame allocation offset existed)
  · intro i
    exact congrArg (interpret M profile ((input.offset i-a.output.tensor.base)%columns normalized)) (inputFrame i)
  · intro i
    exact congrArg (interpret M profile ((weight.offset i-a.output.tensor.base)%columns normalized)) (weightFrame i)

theorem host_outcomes (M : NativeModel α) (profile : ReductionFamily) (eps : ℝ)
    (rt : Runtime (Term α) input weight normalized (entryGraph entry) preexisting)
    (domain : InputDomain input weight normalized preexisting) :
    HostSpec M profile rt (executeHost M profile eps rt) := by
  have original := Whole.host_outcomes graphAlgebra eps rt domain
  rcases original with failure | failure | failure | failure | success
  · rcases failure with ⟨reason,hx,result⟩
    exact Or.inl ⟨reason,hx,by simp only [executeHost,result,projectOutcome]⟩
  · rcases failure with ⟨xc,reason,hx,hw,result,frame⟩
    refine Or.inr (Or.inl ⟨xc,reason,hx,hw,by simp only [executeHost,result,projectOutcome],?_⟩)
    intro allocation offset existed
    exact congrArg (interpret M profile ((offset-0)%columns normalized)) (frame allocation offset existed)
  · rcases failure with ⟨xc,wc,reason,hx,hw,ha,result,frame⟩
    refine Or.inr (Or.inr (Or.inl ⟨xc,wc,reason,hx,hw,ha,by simp only [executeHost,result,projectOutcome],?_⟩))
    intro allocation offset existed
    exact congrArg (interpret M profile ((offset-0)%columns normalized)) (frame allocation offset existed)
  · rcases failure with ⟨xc,wc,output,reason,hx,hw,ha,hs,result,frame⟩
    refine Or.inr (Or.inr (Or.inr (Or.inl ⟨xc,wc,output,reason,hx,hw,ha,hs,
      by simp only [executeHost,result,projectOutcome],?_⟩)))
    intro allocation offset existed
    exact congrArg (interpret M profile ((offset-output.tensor.base)%columns normalized)) (frame allocation offset existed)
  · rcases success with ⟨xc,wc,output,schedule,final,hx,hw,ha,hs,result,spec⟩
    exact Or.inr (Or.inr (Or.inr (Or.inr ⟨xc,wc,output,schedule,final,hx,hw,ha,hs,
      by simp only [executeHost,result,projectOutcome],returned_projection M profile eps _ domain _ spec⟩)))

def FiniteHostSpec (M : NativeModel α) (profile : ReductionFamily)
    (rt : Runtime (Term α) input weight normalized (entryGraph entry) preexisting)
    (admitted : RuntimeAdmission rt) (outcome : Outcome α) : Prop :=
  HostSpec M profile rt outcome ∧
  ∀ xc wc output schedule (hx : rt.copyInput = .ok xc) (hw : rt.copyWeight xc = .ok wc)
    (ha : rt.allocate xc wc = .ok output) (hs : rt.setup ⟨xc,wc,output⟩ = .ok schedule),
    IndexCorrespondence (Prepared.mk xc wc output) normalized (admitted xc wc output schedule hx hw ha hs)

theorem finite_host_outcomes (M : NativeModel α) (profile : ReductionFamily) (eps : ℝ)
    (rt : Runtime (Term α) input weight normalized (entryGraph entry) preexisting)
    (domain : InputDomain input weight normalized preexisting) (admitted : RuntimeAdmission rt) :
    FiniteHostSpec M profile rt admitted (executeHost M profile eps rt) := by
  exact ⟨host_outcomes M profile eps rt domain,fun xc wc output schedule hx hw ha hs =>
    index_correspondence _ _ (admitted xc wc output schedule hx hw ha hs)⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed
