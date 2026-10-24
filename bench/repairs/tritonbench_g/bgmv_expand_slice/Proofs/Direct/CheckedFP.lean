import bench.repairs.tritonbench_g.bgmv_expand_slice.Contract
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.FPGrid
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.FPScalarTree

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.CheckedFP
open VeriTile Triton FP.Structural FPValues
noncomputable section
variable {α : Type} [Inhabited α]

theorem complete (x w y i : Contract.TensorInfo) (scalars : Contract.Scalars) (indices : Nat → Int)
    (ok : Contract.Admitted x w y i scalars indices) (M : Algebra α) (plans : FPScalarTree.Plans)
    (data : Data α) (metadata : ∀ b,b < i.numel → data.indices b=indices b) :
    let h:=Contract.call x w y i scalars indices ok
    let reduction:=FPScalarTree.reduction M plans
    FPGrid.run M reduction h (Geometry.programs h) data=some (FPGrid.fold M reduction h (Geometry.programs h) data) ∧
    (∀ p : Geometry.Program h,indices p.batch≠-1 →
      (FPGrid.fold M reduction h (Geometry.programs h) data).output (FPGrid.address h p)=FPGrid.value M reduction h p data) ∧
    (∀ offset,(∀ p∈Geometry.programs h,offset≠FPGrid.address h p) →
      (FPGrid.fold M reduction h (Geometry.programs h) data).output offset=data.output offset) := by
  dsimp only
  let h:=Contract.call x w y i scalars indices ok
  let reduction:=FPScalarTree.reduction M plans
  by_cases empty : h.N=0
  · have noProgram : Geometry.programs h=[] := by
      apply List.eq_nil_iff_forall_not_mem.mpr
      intro p _
      have hn : p.row<h.N := p.rowInside
      omega
    refine ⟨by change FPGrid.run M reduction h (Geometry.programs h) data=some (FPGrid.fold M reduction h (Geometry.programs h) data);rw [noProgram];rfl,?_,by intro offset _;change (FPGrid.fold M reduction h (Geometry.programs h) data).output offset=data.output offset;rw [noProgram];rfl⟩
    intro p
    have hn : p.row<h.N := p.rowInside
    omega
  · have valid : ∀ b,b<h.batches → data.indices b=-1 ∨ 0≤data.indices b := by
      intro b hb
      rw [metadata b hb]
      exact (Contract.selected_valid x w y i scalars indices ok b 0 hb (by change 0<h.N;omega)).imp_right And.left
    refine ⟨FPGrid.executes M reduction h (Contract.primary x w y i scalars indices ok) _ data valid,?_,FPGrid.frame M reduction h _ data⟩
    intro p active
    apply FPGrid.values M reduction h _ _ data p (Geometry.programs_complete h p)
    · rw [metadata p.batch p.batchInside];exact active
    · simpa [FPGrid.address,Geometry.outputOffset] using Geometry.programs_unique h

#axiomsClean complete
#specNonCircular FPGrid.value avoiding [FPGrid.run,FPGrid.fold,Kernels.source,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.CheckedFP
