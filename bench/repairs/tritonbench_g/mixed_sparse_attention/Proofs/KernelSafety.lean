import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.MixedSparseAttention.KernelSafety
open VeriTile Triton FP.Structural
open Whole002 Source Whole Checked KernelTensor
noncomputable section
set_option maxHeartbeats 4000000

def capacity (c : Config) (layout : Reify.Layout c) (r : RegionName) : Nat :=
  if r=c.out then tensorSize c else Reify.capacity c layout r
theorem capacity_input (layout : Reify.Layout c) (r : Role) (hr : r∈inputRoles) :
    capacity c layout (r.region c)=r.size c := by
  simp [capacity,input_apart layout r hr,Reify.capacity,layout.inputs r hr]
theorem storage (c : Config) (layout : Reify.Layout c) : Access.Storage c (capacity c layout) := by
  constructor
  · intro r hr
    simp only [List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with rfl|rfl|rfl|rfl
    · exact le_of_eq (capacity_input layout .query (by simp [inputRoles])).symm
    · exact le_of_eq (capacity_input layout .key (by simp [inputRoles])).symm
    · exact le_of_eq (capacity_input layout .value (by simp [inputRoles])).symm
    · simp [capacity]
  · exact le_of_eq (capacity_input layout .sequence (by simp [inputRoles])).symm
  · exact le_of_eq (capacity_input layout .blocks (by simp [inputRoles])).symm
  · exact le_of_eq (capacity_input layout .counts (by simp [inputRoles])).symm
  · exact le_of_eq (capacity_input layout .offsets (by simp [inputRoles])).symm
  · exact le_of_eq (capacity_input layout .columns (by simp [inputRoles])).symm
theorem base_eq (c : Config) (request : Request) (layout : Reify.Layout c) (r : Role) :
    Reify.base c request layout (r.region c)=(request.tensor r).pointer := by
  by_cases hr : r=.output
  · subst r;simp [Reify.base,Role.region,layout.output]
  · have hi : r∈inputRoles := by cases r <;> simp_all [inputRoles]
    simp [Reify.base,layout.inputs r hi]
theorem machine (c : Config) (request : Request) (layout : Reify.Layout c)
    (accepted : Admission c request values) : Machine.Closed c (Reify.base c request layout) := by
  apply Machine.closed
  refine ⟨accepted.envelope,?_⟩
  intro r width hw
  simp only [Machine.bindings,List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at hw
  rcases hw with h|h|h|h|h|h|h|h|h
  all_goals rcases h with ⟨rfl,rfl⟩
  all_goals first
    | simpa only [←base_eq c request layout .query] using accepted.pointers .query
    | simpa only [←base_eq c request layout .key] using accepted.pointers .key
    | simpa only [←base_eq c request layout .value] using accepted.pointers .value
    | simpa only [←base_eq c request layout .output] using accepted.pointers .output
    | simpa only [←base_eq c request layout .sequence] using accepted.pointers .sequence
    | simpa only [←base_eq c request layout .blocks] using accepted.pointers .blocks
    | simpa only [←base_eq c request layout .offsets] using accepted.pointers .offsets
    | simpa only [←base_eq c request layout .counts] using accepted.pointers .counts
    | simpa only [←base_eq c request layout .columns] using accepted.pointers .columns

theorem source_safe {α β : Type} [Inhabited α] (c : Config) (request : Request) (layout : Reify.Layout c)
    (read : Nat → β) (zero : β) (encode : β → α) (integer : β → Int)
    (accepted : Admission c request (observed c request read zero integer)) :
    Access.Safe c (capacity c layout) (initial c request layout read zero encode integer) ∧
      Machine.Closed c (Reify.base c request layout) :=
  ⟨Access.safe c accepted.shape _ (storage c layout) _
    (Checked.metadata c _ _ accepted.metadata (metadata_initial layout read zero encode integer)),
   machine c request layout accepted⟩

#axiomsClean storage
#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.MixedSparseAttention.KernelSafety
