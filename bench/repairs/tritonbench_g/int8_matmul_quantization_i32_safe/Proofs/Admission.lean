import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ComposedTensor

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Admission
open VeriTile.Triton FP.Structural Direct
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {F H : Type} [Inhabited F]

/-- Physical Tensor admission uses the existing layout/pointer laws and
allocation API. These fields constrain metadata/identity, not values/results. -/
structure Gemm (base : Nat → Nat) (request : Tensor.Request) (caller : Memory F H) (live : Nat → Prop)
    (alloc : GemmAllocation request.original caller live) : Prop where
  inputs : request.original.live live
  laws : GemmLaws base request.original
  output : TensorLaws base alloc.output

theorem separate (base : Nat → Nat) (request : Tensor.Request) (caller : Memory F H) (live : Nat → Prop)
    (alloc : GemmAllocation request.original caller live) (admit : Gemm base request caller live alloc) :
    (request.original.buffers alloc.output.info.storage).Separate :=
  alloc.separate request.original caller live admit.inputs

/-- The source's prepared ABI state reads exactly the actual post-allocation
backing. A snapshot cannot hide a mutation or an output/input alias. -/
theorem prepared_binding (pack : H → F) (base : Nat → Nat) (request : Tensor.Request)
    (caller : Memory F H) (live : Nat → Prop) (alloc : GemmAllocation request.original caller live)
    (admit : Gemm base request caller live alloc) (template : State F) :
    Tensor.bind pack request.original alloc.output.info.storage caller alloc.memory template =
      Tensor.bind pack request.original alloc.output.info.storage alloc.memory alloc.memory template := by
  have same:=Tensor.allocated_inputs request.original caller live admit.inputs alloc
  have ha:=same request.original.a (by simp [GemmRequest.views])
  have hb:=same request.original.b (by simp [GemmRequest.views])
  have has:=same request.original.aScale (by simp [GemmRequest.views])
  have hbs:=same request.original.bScale (by simp [GemmRequest.views])
  unfold Tensor.bind
  congr 1
  funext region i
  split_ifs <;> simp only [ha,hb,has,hbs]

theorem input_pointers (base : Nat → Nat) (request : Tensor.Request) (caller : Memory F H) (live : Nat → Prop)
    (alloc : GemmAllocation request.original caller live) (admit : Gemm base request caller live alloc) :
    (∀ t∈request.original.views, InputPointerBinding base t) ∧ InputPointerBinding base alloc.output := by
  exact ⟨fun t ht => input_pointer_binding base t (admit.laws.inputs t ht),input_pointer_binding base alloc.output admit.output⟩

theorem supplied_identity (r : Direct.GemmRequest) (caller : Memory F H) (live : Nat → Prop)
    (alloc : GemmAllocation r caller live) (out : TensorView) (supplied : r.out=some out) : alloc.output=out := by
  simpa [GemmDescriptor,supplied] using alloc.descriptor

/-- Composed GEMM sees the original live storages and both actual producer
allocations. Fresh output allocation consequently cannot overlap any input. -/
def composedLive (r : ComposedRequest) (caller : Memory F H) (live : Nat → Prop)
    (alloc : RowAllocation r.row caller live) (storage : Nat) : Prop :=
  live storage ∨ storage=alloc.data.info.storage ∨ storage=alloc.scales.info.storage

theorem composed_inputs (r : ComposedRequest)
    (safe : LaunchSafe (rows r.fpa.info) (cols r.b.info) (cols r.fpa.info))
    (caller : Memory F H) (live : Nat → Prop) (inputs : r.live live) (alloc : RowAllocation r.row caller live) :
    (ProducerTensor.producedRequest r safe caller live inputs alloc).original.live (composedLive r caller live alloc) := by
  intro t ht
  have identities:=Direct.producedRequest_original r caller live inputs alloc
  change t∈(Direct.producedRequest r caller live inputs alloc).views at ht
  simp only [GemmRequest.views,List.mem_cons,List.not_mem_nil,or_false] at ht
  rcases ht with rfl|rfl|rfl|rfl
  · right;left;rw [identities.1]
  · right;right;rw [identities.2]
  · left;exact inputs r.b (by simp [ComposedRequest.views])
  · left;exact inputs r.bScale (by simp [ComposedRequest.views])

theorem producer_prepared_binding (pack : H → F) (r : RowRequest) (caller : Memory F H) (live : Nat → Prop)
    (input : live r.input.info.storage) (alloc : RowAllocation r caller live) (template : State F) :
    ProducerTensor.bind pack r caller template=ProducerTensor.bind pack r alloc.memory template := by
  have sep:=alloc.separate r caller live input
  have same : ∀ i, alloc.memory (r.input.info.storage,i)=caller (r.input.info.storage,i) := by
    intro i
    exact alloc.frame _ sep.1 sep.2.1
  unfold ProducerTensor.bind
  congr 1
  funext region i
  split_ifs <;> simp only [same]

/-- Standalone producer and Torch helper admission remain exactly the
shared frozen contracts, independent of the active GEMM K restriction. -/
abbrev RowContract := Direct.RowContract
abbrev HelperContract := Direct.HelperContract

#axiomsClean separate
#axiomsClean prepared_binding
#axiomsClean input_pointers
#axiomsClean supplied_identity
#axiomsClean composed_inputs
#axiomsClean producer_prepared_binding
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Admission
