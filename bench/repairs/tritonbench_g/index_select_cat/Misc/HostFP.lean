import bench.repairs.tritonbench_g.index_select_cat.Proofs.Physical

namespace VeriTile.Bench.Repairs.IndexSelectCat.FPEquiv
open VeriTile.Triton FP.Structural Physical
open VeriTile.Torch.FP32
noncomputable section
local instance : Inhabited Physical.Word := ⟨FP.Value.zero .fp32⟩

/-- Exact complete FP32 Tensor result and full caller-byte frame. The source
copies all FP32 words without floating operations. The int64 binding is the
ordinary same-input CPU/native-load observation, never an output premise. -/
theorem whole_tensor_fp32 (rest : Algebra Physical.Word) (r : Request)
    (checked : checkInputs r=Bool.true) (storage : StorageBinding r) (objects : Objects r)
    (m : Memory) (ids : IndexBinding r m) (incoming : VeriTile.Torch.Device) :
    let finalMemory := Physical.run rest r m objects.output.object incoming
    (∀ i j,i<r.N → j<r.D →
      logicalValue (read finalMemory) objects.output objects.outputLaws (i*r.D+j)=
        Host.target r.indices
          (fun row col=>logicalValue (read m) objects.source objects.sourceLaws (row*r.D+col)) i j) ∧
    (∀ p,(p<r.output.metadata.dataPtr.toNat ∨ r.output.metadata.dataPtr.toNat+4*(r.N*r.D) ≤ p) →
      finalMemory p=m p) ∧
    (∀ i,i<r.N → indexAt r finalMemory i=r.indices i) ∧
    (Host.run rest r (words m r.source) (words m r.output) objects.output.object incoming).returnedObject=objects.output.object ∧
    (Host.run rest r (words m r.source) (words m r.output) objects.output.object incoming).device=incoming := by
  refine ⟨?_,?_,?_,rfl,rfl⟩
  · intro i j hi hj
    exact logical_readback rest r checked objects m incoming i j hi hj
  · intro p outside
    exact frame rest r m objects.output.object incoming p outside
  · intro i hi
    exact (index_preserved rest r checked storage m objects.output.object incoming i hi).trans (ids i hi).symm

#axiomsClean whole_tensor_fp32
#specNonCircular Host.target avoiding [Physical.run,Host.run,Program.run,TypedGather.execution]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IndexSelectCat.FPEquiv
