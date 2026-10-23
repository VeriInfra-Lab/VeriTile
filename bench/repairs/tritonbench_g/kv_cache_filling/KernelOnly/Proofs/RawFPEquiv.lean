import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.SharedReadback
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.QuantNative

namespace VeriTile.Bench.Repairs.KvCacheFilling.RawNative
open VeriTile.Triton FP.Structural
noncomputable section

local instance : Inhabited NativeProfile.Word := ⟨FP.Value.zero .fp32⟩

/-- The independent native row formula reads only the raw initial kernel
regions. Request contains dimensions and decoded integer scheduling arrays. -/
def kernelNativeTarget (a : NativeProfile.Arithmetic) (reductions : QuantNative.Reductions)
    (r : Request) (initial : Job → State NativeProfile.Word) (j : Job) (address : Address) :=
  SharedTarget.value (QuantNative.ops a reductions) r
    (QuantExecution.inputRow (Shared.parameters r) (initial j) (sourceRow r j) Bool.true)
    (QuantExecution.inputRow (Shared.parameters r) (initial j) (sourceRow r j) Bool.false) address

/-- Complete source and candidate execution for every raw scheduled job, all
independent native outputs, the regional frame, and identical observable
stores. No Tensor metadata, admission checker or constructed post-heap is an
input premise. Scalar and ordered-reduction models are unchanged. -/
theorem kernel_native (a : NativeProfile.Arithmetic) (reductions : QuantNative.Reductions)
    (r : Request) (d : Domain r) (initial : Job → State NativeProfile.Word)
    (src : ∀ j ∈ jobs r,
      ((initial j).mem "src" ((initial j).pids 0)).read .int = (sourceRow r j : Int))
    (dst : ∀ j ∈ jobs r,
      ((initial j).mem "dst" ((initial j).pids 0)).read .int = (destinationRow r j : Int)) :
    ∃ states : Bool → Job → State NativeProfile.Word,
      (∀ version j, j ∈ jobs r →
        exec (QuantProfile.algebra (QuantNative.ops a reductions)) (Shared.kernel version r)
          (initial j) = some (states version j)) ∧
      (∀ version j, j ∈ jobs r → ∀ reg i,
        SharedReadback.Outside r (destinationRow r j) reg i →
          (states version j).mem reg i = (initial j).mem reg i) ∧
      (∀ version j, j ∈ jobs r → ∀ address, address ∈ rowAddresses r j →
        SourceBacking.project r (states version j) address =
          kernelNativeTarget a reductions r initial j address) ∧
      SourceBacking.stores r (states Bool.false) = SourceBacking.stores r (states Bool.true) := by
  let ops := QuantNative.ops a reductions
  let states : Bool → Job → State NativeProfile.Word := fun version j =>
    if hj : j ∈ jobs r then
      (SharedExecution.executed ops r d (initial j) (sourceRow r j)
        (destinationRow r j) (src j hj) (dst j hj) version).val
    else initial j
  have ran : ∀ version j, j ∈ jobs r →
      exec (QuantProfile.algebra ops) (Shared.kernel version r) (initial j) =
        some (states version j) := by
    intro version j hj
    simp only [states,hj,↓reduceDIte]
    exact (SharedExecution.executed ops r d (initial j) (sourceRow r j)
      (destinationRow r j) (src j hj) (dst j hj) version).property
  have values : ∀ version j, j ∈ jobs r → ∀ address, address ∈ rowAddresses r j →
      SourceBacking.project r (states version j) address =
        kernelNativeTarget a reductions r initial j address := by
    intro version j hj address ha
    exact SharedReadback.project ops r d (initial j) (states version j) j
      (src j hj) (dst j hj) version (ran version j hj) address ha
  refine ⟨states,ran,?_,values,?_⟩
  · intro version j hj reg i outside
    exact SharedReadback.frame ops r d (initial j) (states version j)
      (sourceRow r j) (destinationRow r j) (src j hj) (dst j hj)
      version (ran version j hj) reg i outside
  · apply List.flatMap_congr
    intro j hj
    apply List.map_congr_left
    intro address ha
    congr 1
    rw [values Bool.false j hj address ha,values Bool.true j hj address ha]

#axiomsClean kernel_native
#specNonCircular kernelNativeTarget avoiding [FP.Structural.exec, Shared.original,
  Shared.candidate, SourceBacking.stores, SharedExecution.programs]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheFilling.RawNative
