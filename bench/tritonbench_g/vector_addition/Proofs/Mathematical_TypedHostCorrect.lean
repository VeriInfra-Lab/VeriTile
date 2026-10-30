import bench.tritonbench_g.vector_addition.Proofs.Mathematical_TypedHostValues

/-! Public source-host outcome contract on the complete seventeen-by-seventeen
dtype domain. Success comes from actual existing grid executions. Its value
domain is explicit; dtype admission does not imply native value-totality. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedHostCorrect
open VeriTile.Triton DTypeScope TypedHost
open HostWrapper (Tensor Result Run Event Failure parameters successTrace cudaGuard add)
noncomputable section
set_option maxHeartbeats 6000000
local instance : Inhabited TypedHost.Word := ⟨FP.Value.zero .fp32⟩

specification vector_addition_typed_host_success
    {σ : Type} (b : Numerics) (rt : Runtime σ)
    (xD yD : DType) (r : Route) (x y output : Tensor)
    (initial allocated : World σ)
    (hx : x.dtype = dtypeName xD) (hy : y.dtype = dtypeName yD)
    (selected : route xD yD = some r)
    (ha : rt.emptyLike x initial = .ok output allocated)
    (hc : cudaGuard x y output = Bool.true)
    (allocation : EmptyLikeContract x y output initial allocated)
    (domain : SourceDomain x y)
    (values : ValueDomain b r (parameters x y output) allocated) :
    ∃ final, add (withLaunch b rt) x y initial =
        ⟨successTrace x y output, .ok output final⟩ ∧
      ReturnedMetadata x output ∧
      project r final = definedExpected b r (parameters x y output) allocated ∧
      Frame (parameters x y output) allocated final ∧
      UnselectedUnchanged r allocated final := by
  obtain ⟨final, execution, metadata, formula, frame, unchanged⟩ :=
    add_typed_correct b rt xD yD r x y output initial allocated hx hy selected ha hc allocation domain values.1
  refine ⟨final, execution, metadata, ?_, frame, unchanged⟩
  exact formula.trans (expected_policy_independent b r (parameters x y output) allocated values.2.2)

def rejectedTrace (x y output : Tensor) : List Event :=
  [.emptyLike x.region, .cudaCheck x.isCUDA y.isCUDA output.isCUDA,
   .numel output.region output.numel,
   .launch output.numel 1024 (ValueProofs.gridCount (parameters x y output))]

/-- Every canonical pair reaches either its first frontend rejection or an
actual typed source grid on the stated value domain. The rejected branches
have no returned-tensor event and do not evaluate execute. -/
specification vector_addition_typed_host_outcome
    {σ : Type} (b : Numerics) (rt : Runtime σ)
    (xD yD : DType) (x y output : Tensor) (initial allocated : World σ)
    (hx : x.dtype = dtypeName xD) (hy : y.dtype = dtypeName yD)
    (ha : rt.emptyLike x initial = .ok output allocated)
    (hc : cudaGuard x y output = Bool.true)
    (allocation : EmptyLikeContract x y output initial allocated)
    (domain : SourceDomain x y)
    (values : ∀ r, route xD yD = some r → ValueDomain b r (parameters x y output) allocated) :
    (∃ arg bad, DTypeAdmission.admission xD yD = .signatureRejected arg bad ∧
      add (withLaunch b rt) x y initial =
        ⟨rejectedTrace x y output, .error (.native "frontend: signature rejected") allocated⟩) ∨
    (DTypeAdmission.admission xD yD = .additionRejected ∧
      add (withLaunch b rt) x y initial =
        ⟨rejectedTrace x y output, .error (.native "frontend: addition rejected") allocated⟩) ∨
    (∃ r final, route xD yD = some r ∧
      add (withLaunch b rt) x y initial = ⟨successTrace x y output, .ok output final⟩ ∧
      ReturnedMetadata x output ∧
      project r final = definedExpected b r (parameters x y output) allocated ∧
      Frame (parameters x y output) allocated final ∧
      UnselectedUnchanged r allocated final) := by
  cases h : DTypeAdmission.admission xD yD with
  | signatureRejected arg bad =>
    exact Or.inl ⟨arg, bad, rfl, signature_rejection b rt xD yD arg bad x y output initial allocated hx hy ha hc h⟩
  | additionRejected =>
    exact Or.inr (Or.inl ⟨rfl, addition_rejection b rt xD yD x y output initial allocated hx hy ha hc h⟩)
  | passesDTypeGates computation store =>
    have existsRoute := route_matches_admission xD yD
    simp only [h, DTypeAdmission.Outcome.stage, decide_true] at existsRoute
    cases hr : route xD yD with
    | none => simp [hr] at existsRoute
    | some r =>
      obtain ⟨final, execution, metadata, formula, frame, unchanged⟩ :=
        vector_addition_typed_host_success b rt xD yD r x y output initial allocated hx hy hr ha hc
          allocation domain (values r hr)
      exact Or.inr (Or.inr ⟨r, final, rfl, execution, metadata, formula, frame, unchanged⟩)

specification vector_addition_typed_host_allocation_error
    {σ : Type} (b : Numerics) (rt : Runtime σ) (x y : Tensor)
    (initial failed : World σ) (reason : Failure)
    (failedAllocation : rt.emptyLike x initial = .error reason failed) :
    add (withLaunch b rt) x y initial = ⟨[.emptyLike x.region], .error reason failed⟩ :=
  HostWrapper.allocation_error_first (withLaunch b rt) x y initial failed reason failedAllocation

specification vector_addition_typed_host_cuda_error
    {σ : Type} (b : Numerics) (rt : Runtime σ) (x y output : Tensor)
    (initial allocated : World σ)
    (ha : rt.emptyLike x initial = .ok output allocated)
    (hc : cudaGuard x y output = Bool.false) :
    add (withLaunch b rt) x y initial =
      ⟨[.emptyLike x.region, .cudaCheck x.isCUDA y.isCUDA output.isCUDA],
       .error .cudaAssertion allocated⟩ :=
  HostWrapper.cuda_error_after_allocation (withLaunch b rt) x y output initial allocated ha hc

#axiomsClean vector_addition_typed_host_success
#axiomsClean vector_addition_typed_host_outcome
#axiomsClean vector_addition_typed_host_allocation_error
#axiomsClean vector_addition_typed_host_cuda_error
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedHostCorrect
