import bench.tritonbench_g.rmsnorm_triton.Proofs.Mathematical_WholeMath
import bench.tritonbench_g.rmsnorm_triton.Proofs.Mathematical_WholeFinite
import bench.tritonbench_g.rmsnorm_triton.Proofs.Mathematical_NativeReduction

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

structure Runtime (α : Type) (p : Parameters) (entry : Memory α) (preexisting : Nat → Prop) where
  allocate : Except String (Allocated α p entry preexisting)
  setup : (a : Allocated α p entry preexisting) → Except String (Schedule p)

inductive Outcome (α : Type) where
  | shapeFailure
  | weightShapeFailure
  | weightWidthFailure
  | allocationFailure (reason : String)
  | weightRankFailure (out : Tensor) (memory : Memory α)
  | runtimeFailure (reason : String) (out : Tensor) (memory : Memory α)
  | kernelFailure
  | returned (out : Tensor) (memory : Memory α)

/-- The assertion precedes allocation. A scalar weight fails shape[-1]; a
nonvector weight with matching final width fails stride expansion after
allocation. The runtime supplies only allocation facts and launch order. -/
def executeHost (Q : Numerics α) (red : Reduction) (p : Parameters) (u : Nat → α)
    (rt : Runtime α p entry preexisting) : Outcome α :=
  if p.input.shape.length = 3 then
    if p.weights.shape = [] then .weightShapeFailure
    else if p.weights.shape.getLast? = some p.width then
      match rt.allocate with
      | .error reason => .allocationFailure reason
      | .ok a =>
        if p.weights.shape.length = 1 then
          match rt.setup a with
          | .error reason => .runtimeFailure reason p.output a.memory
          | .ok schedule =>
            match entryRun Q red p a u schedule with
            | none => .kernelFailure
            | some final => .returned p.output final
        else .weightRankFailure p.output a.memory
    else .weightWidthFailure
  else .shapeFailure

def ReturnedSpec (p : Parameters) (formula : Nat → Nat → Nat → α)
    (a : Allocated α p entry preexisting) (final : Memory α) : Prop :=
  p.output.shape = p.input.shape ∧ p.output.dtype = p.input.dtype ∧
  p.output.device = p.input.device ∧
  (∀ b m k, b < p.batches → m < p.rows → k < p.width →
    final p.output.allocation (outputOffset p b m k) = formula b m k) ∧
  (∀ allocation offset, Outside p allocation offset → final allocation offset = a.memory allocation offset) ∧
  (∀ allocation offset, preexisting allocation → final allocation offset = entry allocation offset)

/-- The full host contract contains each original failure phase, return metadata,
all logical output values, the exterior frame and every preexisting cell. -/
def HostSpec (p : Parameters) (formula : Nat → Nat → Nat → α)
    (rt : Runtime α p entry preexisting) (result : Outcome α) : Prop :=
  if p.input.shape.length = 3 then
    if p.weights.shape = [] then result = .weightShapeFailure
    else if p.weights.shape.getLast? = some p.width then
      (∃ reason, rt.allocate = .error reason ∧ result = .allocationFailure reason) ∨
      (∃ a, rt.allocate = .ok a ∧ p.weights.shape.length ≠ 1 ∧
        result = .weightRankFailure p.output a.memory ∧
        ∀ allocation offset, preexisting allocation → a.memory allocation offset = entry allocation offset) ∨
      (∃ a reason, rt.allocate = .ok a ∧ p.weights.shape.length = 1 ∧
        rt.setup a = .error reason ∧ result = .runtimeFailure reason p.output a.memory ∧
        ∀ allocation offset, preexisting allocation → a.memory allocation offset = entry allocation offset) ∨
      (∃ a schedule final, rt.allocate = .ok a ∧ p.weights.shape.length = 1 ∧
        rt.setup a = .ok schedule ∧ result = .returned p.output final ∧ ReturnedSpec p formula a final)
    else result = .weightWidthFailure
  else result = .shapeFailure

def logicalTensor (Q : Numerics α) (red : Reduction) (p : Parameters) (entry : Memory α)
    (b m k : Nat) : α :=
  logicalValue Q red p.width p.eps
    (fun j => entry p.input.allocation (inputOffset p b m j))
    (fun j => entry p.weights.allocation (weightOffset p j)) k

def mathematicalTensor (p : Parameters) (entry : Memory ℝ) (b m k : Nat) : ℝ :=
  mathematical p.width p.eps
    (fun j => entry p.input.allocation (inputOffset p b m j))
    (fun j => entry p.weights.allocation (weightOffset p j)) k

def ValidRuntime (p : Parameters) (rt : Runtime α p entry preexisting) : Type :=
  ∀ a, rt.allocate = .ok a → p.input.shape.length = 3 →
    p.weights.shape.getLast? = some p.width → p.weights.shape.length = 1 → Domain p

theorem host_outcomes (Q : Numerics α) (red : Reduction) (p : Parameters) (u : Nat → α)
    (rt : Runtime α p entry preexisting) (valid : ValidRuntime p rt) :
    HostSpec p (logicalTensor Q red p entry) rt (executeHost Q red p u rt) := by
  by_cases hx : p.input.shape.length = 3
  · by_cases he : p.weights.shape = []
    · simp only [HostSpec,executeHost,hx,he,if_true]
    · by_cases hw : p.weights.shape.getLast? = some p.width
      · simp only [HostSpec,hx,he,hw,if_true,if_false]
        cases ha : rt.allocate with
        | error reason => exact Or.inl ⟨reason,rfl,by simp [executeHost,hx,he,hw,ha]⟩
        | ok a =>
          by_cases hr : p.weights.shape.length = 1
          · cases hs : rt.setup a with
            | error reason =>
              exact Or.inr (Or.inr (Or.inl
                ⟨a,reason,rfl,hr,hs,by simp [executeHost,hx,he,hw,ha,hr,hs],a.frame⟩))
            | ok schedule =>
              obtain ⟨final,ran,values,frame,entryFrame⟩ := entry_result Q red p a (valid a ha hx hw hr) u schedule
              exact Or.inr (Or.inr (Or.inr
                ⟨a,schedule,final,rfl,hr,hs,by simp [executeHost,hx,he,hw,ha,hr,hs,ran],
                  a.outputShape,a.outputDtype,a.outputDevice,values,frame,entryFrame⟩))
          · exact Or.inr (Or.inl
              ⟨a,rfl,hr,by simp [executeHost,hx,he,hw,ha,hr],a.frame⟩)
      · simp only [HostSpec,executeHost,hx,he,hw,if_true,if_false]
  · simp only [HostSpec,executeHost,hx,if_false]

theorem real_host_outcomes (red : Reduction) (p : Parameters) (u : Nat → ℝ)
    (rt : Runtime ℝ p entry preexisting) (valid : ValidRuntime p rt) :
    HostSpec p (mathematicalTensor p entry) rt (executeHost realNumerics red p u rt) := by
  have equal : logicalTensor realNumerics red p entry = mathematicalTensor p entry := by
    funext b m k
    exact logicalValue_real red p.width p.eps _ _ k
  simpa only [equal] using host_outcomes realNumerics red p u rt valid

def RuntimeAdmission (p : Parameters) (rt : Runtime α p entry preexisting)
    (valid : ValidRuntime p rt) : Prop :=
  ∀ a (ha : rt.allocate = .ok a) (hx : p.input.shape.length = 3)
    (hw : p.weights.shape.getLast? = some p.width) (hr : p.weights.shape.length = 1),
    Nonempty (FiniteAdmission p (valid a ha hx hw hr) preexisting)

def FiniteHostSpec (p : Parameters) (formula : Nat → Nat → Nat → α)
    (rt : Runtime α p entry preexisting) (valid : ValidRuntime p rt)
    (result : Outcome α) : Prop :=
  HostSpec p formula rt result ∧
  ∀ a (ha : rt.allocate = .ok a) (hx : p.input.shape.length = 3)
    (hw : p.weights.shape.getLast? = some p.width) (hr : p.weights.shape.length = 1),
    ∃ f : FiniteAdmission p (valid a ha hx hw hr) preexisting,
      IndexCorrespondence p (valid a ha hx hw hr) f

theorem finite_host_outcomes (Q : Numerics α) (red : Reduction) (p : Parameters) (u : Nat → α)
    (rt : Runtime α p entry preexisting) (valid : ValidRuntime p rt)
    (admitted : RuntimeAdmission p rt valid) :
    FiniteHostSpec p (logicalTensor Q red p entry) rt valid (executeHost Q red p u rt) := by
  refine ⟨host_outcomes Q red p u rt valid,?_⟩
  intro a ha hx hw hr
  obtain ⟨f⟩ := admitted a ha hx hw hr
  exact ⟨f,index_correspondence p _ f⟩

theorem finite_real_host_outcomes (red : Reduction) (p : Parameters) (u : Nat → ℝ)
    (rt : Runtime ℝ p entry preexisting) (valid : ValidRuntime p rt)
    (admitted : RuntimeAdmission p rt valid) :
    FiniteHostSpec p (mathematicalTensor p entry) rt valid (executeHost realNumerics red p u rt) := by
  refine ⟨real_host_outcomes red p u rt valid,?_⟩
  intro a ha hx hw hr
  obtain ⟨f⟩ := admitted a ha hx hw hr
  exact ⟨f,index_correspondence p _ f⟩

#guard_msgs (drop info) in
#specNonCircular mathematical avoiding [executeHost,entryRun,executeGrid,executeRow,outputPass]
#guard_msgs (drop info) in
#specNonCircular logicalValue avoiding [executeHost,entryRun,executeGrid,executeRow,outputPass]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
