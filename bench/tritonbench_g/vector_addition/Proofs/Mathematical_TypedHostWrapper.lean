import bench.tritonbench_g.vector_addition.Proofs.Mathematical_TypedHostExecution

namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedHost
open VeriTile.Triton DTypeScope
open HostWrapper (Tensor Result Run Event Failure Primitive parameters successTrace cudaGuard add)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

abbrev Runtime (σ : Type) := HostWrapper.Runtime Word (Native σ)

/-- Unknown API dtypes and mathematical interpreter faults are explicitly
logical diagnostics. The two native-tagged rejection cases represent the
trusted frontend primitive classification on the frozen canonical domain;
their strings are category labels, not claims about diagnostic spelling. -/
def launch {σ : Type} (b : Numerics) (x y _output : Tensor)
    (p : ValueProofs.Parameters) (s : World σ) : Result (World σ) Unit :=
  match parseDType x.dtype, parseDType y.dtype with
  | some xD, some yD => match DTypeAdmission.admission xD yD with
    | .signatureRejected _ _ => .error (.native "frontend: signature rejected") s
    | .additionRejected => .error (.native "frontend: addition rejected") s
    | .passesDTypeGates _ _ => match route xD yD with
      | none => .error .logicalInterpreter s
      | some r => match execute b r p s with
        | .ok t => .ok () t
        | .error _ => .error .logicalInterpreter s
  | _, _ => .error .logicalInterpreter s

def withLaunch {σ : Type} (b : Numerics) (rt : Runtime σ) : Runtime σ :=
  { rt with launch := launch b }

/-- The existing source skeleton already executes allocation, then CUDA
assertion, numel, launch and return. This lemma only composes those calls. -/
theorem add_success {σ : Type} (rt : Runtime σ) (x y output : Tensor)
    (initial allocated final : World σ)
    (ha : rt.emptyLike x initial = .ok output allocated)
    (hc : cudaGuard x y output = Bool.true)
    (hl : rt.launch x y output (parameters x y output) allocated = .ok () final) :
    add rt x y initial = ⟨successTrace x y output, .ok output final⟩ := by
  simp [add, HostWrapper.perform, HostWrapper.Run.bind, ha, hc, hl, successTrace]

theorem launch_selected {σ : Type} (b : Numerics) (xD yD : DType) (r : Route)
    (x y output : Tensor) (p : ValueProofs.Parameters) (s t : World σ)
    (hx : x.dtype = dtypeName xD) (hy : y.dtype = dtypeName yD)
    (selected : route xD yD = some r) (executed : execute b r p s = .ok t) :
    launch b x y output p s = .ok () t := by
  have h := route_matches_admission xD yD
  rw [selected] at h
  have stage : (DTypeAdmission.admission xD yD).stage = .admitted := by
    exact of_decide_eq_true h.symm
  cases ha : DTypeAdmission.admission xD yD with
  | signatureRejected arg dtype => simp [ha, DTypeAdmission.Outcome.stage] at stage
  | additionRejected => simp [ha, DTypeAdmission.Outcome.stage] at stage
  | passesDTypeGates computation store =>
    simp [launch, hx, hy, parse_dtype, ha, selected, executed]

structure ReturnedMetadata (x output : Tensor) : Prop where
  dtype : output.dtype = x.dtype
  shape : output.shape = x.shape
  numel : output.numel = x.numel
  cuda : output.isCUDA = x.isCUDA
  device : output.deviceIndex = x.deviceIndex
  storageOffset : output.storageOffset = 0

/-- A concrete trusted empty_like contract: returned metadata, fresh output,
and preservation of all previously existing non-output cells in both carriers.
Output bytes/cells are arbitrary. Strides and actual pointer-view validity are
separate applicability obligations; empty_like does not copy input contents. -/
structure EmptyLikeContract {σ : Type} (x y output : Tensor)
    (initial allocated : World σ) : Prop where
  metadata : ReturnedMetadata x output
  freshX : x.region ≠ output.region
  freshY : y.region ≠ output.region
  sharedFrame : ∀ r i, r ≠ output.region → allocated.kernel.mem r i = initial.kernel.mem r i
  wideFrame : ∀ r i, r ≠ output.region.name → allocated.native.wide r i = initial.native.wide r i

def SourceDomain (x y : Tensor) : Prop := x.numel < 2^31 ∧ x.numel ≤ y.numel

theorem allocation_preserves_inputs {σ : Type} (x y output : Tensor)
    (initial allocated : World σ) (allocation : EmptyLikeContract x y output initial allocated) :
    (∀ i, allocated.kernel.mem x.region i = initial.kernel.mem x.region i ∧
      allocated.kernel.mem y.region i = initial.kernel.mem y.region i) ∧
    (∀ i, allocated.native.wide x.region.name i = initial.native.wide x.region.name i ∧
      allocated.native.wide y.region.name i = initial.native.wide y.region.name i) := by
  refine ⟨fun i => ⟨allocation.sharedFrame _ i allocation.freshX,
    allocation.sharedFrame _ i allocation.freshY⟩, ?_⟩
  intro i
  have hx : x.region.name ≠ output.region.name := by
    intro h
    apply allocation.freshX
    cases hx : x.region
    cases ho : output.region
    simpa [hx, ho] using h
  have hy : y.region.name ≠ output.region.name := by
    intro h
    apply allocation.freshY
    cases hy : y.region
    cases ho : output.region
    simpa [hy, ho] using h
  exact ⟨allocation.wideFrame _ i hx, allocation.wideFrame _ i hy⟩

/-- No launch success or final memory is assumed. The selected existing grid
constructs both, then the original host skeleton returns the allocated tensor.
The precise OutputDefined premise for wide integer outputs remains inside Ready. -/
theorem add_typed_correct {σ : Type} (b : Numerics) (rt : Runtime σ)
    (xD yD : DType) (r : Route) (x y output : Tensor)
    (initial allocated : World σ)
    (hx : x.dtype = dtypeName xD) (hy : y.dtype = dtypeName yD)
    (selected : route xD yD = some r)
    (ha : rt.emptyLike x initial = .ok output allocated)
    (hc : cudaGuard x y output = Bool.true)
    (allocation : EmptyLikeContract x y output initial allocated)
    (_domain : SourceDomain x y)
    (ready : Ready b r (parameters x y output) allocated) :
    ∃ final, add (withLaunch b rt) x y initial =
        ⟨successTrace x y output, .ok output final⟩ ∧
      ReturnedMetadata x output ∧
      project r final = expected b r (parameters x y output) allocated ∧
      Frame (parameters x y output) allocated final ∧
      UnselectedUnchanged r allocated final := by
  obtain ⟨final, executed, formula, frame, unchanged⟩ :=
    execute_correct b r (parameters x y output) allocated rfl ready
  refine ⟨final, ?_, allocation.metadata, formula, frame, unchanged⟩
  apply add_success
  · exact ha
  · exact hc
  · exact launch_selected b xD yD r x y output (parameters x y output) allocated final hx hy selected executed

theorem signature_rejection {σ : Type} (b : Numerics) (rt : Runtime σ)
    (xD yD : DType) (arg : DTypeAdmission.Argument) (bad : DType)
    (x y output : Tensor) (initial allocated : World σ)
    (hx : x.dtype = dtypeName xD) (hy : y.dtype = dtypeName yD)
    (ha : rt.emptyLike x initial = .ok output allocated)
    (hc : cudaGuard x y output = Bool.true)
    (rejected : DTypeAdmission.admission xD yD = .signatureRejected arg bad) :
    add (withLaunch b rt) x y initial =
      ⟨[.emptyLike x.region, .cudaCheck x.isCUDA y.isCUDA output.isCUDA,
         .numel output.region output.numel,
         .launch output.numel 1024 (ValueProofs.gridCount (parameters x y output))],
       .error (.native "frontend: signature rejected") allocated⟩ := by
  apply HostWrapper.launch_error_after_metadata
  · exact ha
  · exact hc
  · simp [withLaunch, launch, hx, hy, parse_dtype, rejected]

theorem addition_rejection {σ : Type} (b : Numerics) (rt : Runtime σ)
    (xD yD : DType) (x y output : Tensor) (initial allocated : World σ)
    (hx : x.dtype = dtypeName xD) (hy : y.dtype = dtypeName yD)
    (ha : rt.emptyLike x initial = .ok output allocated)
    (hc : cudaGuard x y output = Bool.true)
    (rejected : DTypeAdmission.admission xD yD = .additionRejected) :
    add (withLaunch b rt) x y initial =
      ⟨[.emptyLike x.region, .cudaCheck x.isCUDA y.isCUDA output.isCUDA,
         .numel output.region output.numel,
         .launch output.numel 1024 (ValueProofs.gridCount (parameters x y output))],
       .error (.native "frontend: addition rejected") allocated⟩ := by
  apply HostWrapper.launch_error_after_metadata
  · exact ha
  · exact hc
  · simp [withLaunch, launch, hx, hy, parse_dtype, rejected]

#axiomsClean add_success
#axiomsClean launch_selected
#axiomsClean allocation_preserves_inputs
#axiomsClean add_typed_correct
#axiomsClean signature_rejection
#axiomsClean addition_rejection
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedHost
