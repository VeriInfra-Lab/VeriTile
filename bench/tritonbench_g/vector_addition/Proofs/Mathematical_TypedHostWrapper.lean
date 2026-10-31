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

/-- The existing source skeleton already executes allocation, then CUDA
assertion, numel, launch and return. This lemma only composes those calls. -/
theorem add_success {σ : Type} (rt : Runtime σ) (x y output : Tensor)
    (initial allocated final : World σ)
    (ha : rt.emptyLike x initial = .ok output allocated)
    (hc : cudaGuard x y output = Bool.true)
    (hl : rt.launch x y output (parameters x y output) allocated = .ok () final) :
    add rt x y initial = ⟨successTrace x y output, .ok output final⟩ := by
  simp [add, HostWrapper.perform, HostWrapper.Run.bind, ha, hc, hl, successTrace]

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

#axiomsClean add_success
#axiomsClean allocation_preserves_inputs
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedHost
