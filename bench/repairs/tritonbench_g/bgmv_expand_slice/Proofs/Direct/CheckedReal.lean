import bench.repairs.tritonbench_g.bgmv_expand_slice.Contract
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.Direct.RealGrid

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.CheckedReal
open VeriTile Triton
noncomputable section

/-- The native wrapper supplies the original input, weight, output and integer
metadata windows. This field is the local integer-load observation only. -/
theorem complete (x w y i : Contract.TensorInfo) (scalars : Contract.Scalars) (indices : Nat → Int)
    (ok : Contract.Admitted x w y i scalars indices) (initial : BlockState)
    (metadata : ∀ b,b < i.numel → initial.readMemValue .int "Indices" b=indices b) :
    let h:=Contract.call x w y i scalars indices ok
    ∃ final,RealGrid.run h (Geometry.programs h) initial=some final ∧
      (∀ p : Geometry.Program h,indices p.batch≠-1 →
        final.readMem "Y" (Geometry.outputOffset h p.batch p.row 0)=
          RealExecution.formula h p initial (indices p.batch).toNat) ∧
      (∀ r off,¬RealGrid.Writes h indices (Geometry.programs h) r off → final.mem r off=initial.mem r off) := by
  dsimp only
  let h:=Contract.call x w y i scalars indices ok
  by_cases empty : h.N=0
  · have noProgram : Geometry.programs h=[] := by
      apply List.eq_nil_iff_forall_not_mem.mpr
      intro p _
      have hn : p.row<h.N := p.rowInside
      omega
    refine ⟨initial,by change RealGrid.run h (Geometry.programs h) initial=some initial;rw [noProgram];rfl,?_,by intros;rfl⟩
    intro p
    have hn : p.row<h.N := p.rowInside
    omega
  · have valid : ∀ b,b<h.batches → indices b=-1 ∨ 0 ≤ indices b := by
      intro b hb
      exact (Contract.selected_valid x w y i scalars indices ok b 0 hb (by change 0<h.N;omega)).imp_right And.left
    obtain ⟨final,run,values,frame⟩:=RealGrid.complete h (Geometry.programs h) (Geometry.programs_unique h)
      indices valid initial metadata
    exact ⟨final,run,fun p active => values p (Geometry.programs_complete h p) active,frame⟩

#axiomsClean complete
#specNonCircular RealExecution.formula avoiding [RealGrid.run,VeriTile.Triton.exec,Kernels.source]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvExpandSlice.Direct.CheckedReal
