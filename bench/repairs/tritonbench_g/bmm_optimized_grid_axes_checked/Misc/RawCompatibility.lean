import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.LegacyIO
import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.PublicResults

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.RawCompatibility
open VeriTile.Triton FP.Structural BmmOptimized
noncomputable section

/-- Complete quoted grid under any legal CTA ordering, with the independent
full-K matrix product and complete tagged-memory frame. All inputs are raw. -/
theorem kernel_real (s : Shape) (c : Config) (rest : Algebra ℝ)
    (ps : List Program) (schedule : ps.Perm (programs s c))
    (before : Memory ℝ) (initial : State ℝ) (related : Bridge.Related before initial) :
    ∃ final,Bridge.grid s c realScalar rest ps initial=some final ∧
      (∀ x,s.Active x → final.mem "O" (s.o x.batch x.row x.column)=
        .mk .real (mathematical s before x)) ∧ Bridge.Frame s initial final := by
  obtain ⟨final,run,values,frame⟩:=LegacyIO.complete s c realScalar rest ps schedule before initial related
  refine ⟨final,run,?_,frame⟩
  intro x hx
  rw [values x hx,ordered_real]

#axiomsClean kernel_real
#specNonCircular mathematical avoiding [Bridge.grid,Execution.exec]
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.RawCompatibility

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.RawCompatibility
open VeriTile.Triton FP.Structural BmmOptimized
noncomputable section

/-- Complete quoted native grid, every raw FP32 output word, and the complete
frame. Only the existing pointwise scalar FMA correspondence is assumed. -/
theorem kernel_native (s : Shape) (c : Config) (actual target : Native.Instructions)
    (fma : ∀ l r carry,actual.fmaRN l r carry=target.fmaRN l r carry)
    (rest : Algebra Native.Word) (ps : List Program) (schedule : ps.Perm (programs s c))
    (before : Memory Native.Word) (initial : State Native.Word)
    (related : Bridge.Related before initial) :
    ∃ final,Bridge.grid s c (Native.scalar actual) rest ps initial=some final ∧
      (∀ x,s.Active x → final.mem "O" (s.o x.batch x.row x.column)=
        .mk .real (ordered s (Native.scalar target) before x)) ∧ Bridge.Frame s initial final := by
  obtain ⟨final,run,values,frame⟩:=LegacyIO.complete s c (Native.scalar actual) rest ps schedule before initial related
  have scalar_eq : Native.scalar actual=Native.scalar target := by
    change Scalar.mk Native.zero actual.fmaRN=Scalar.mk Native.zero target.fmaRN
    congr 1
    funext l r carry
    exact fma l r carry
  refine ⟨final,run,?_,frame⟩
  simpa only [scalar_eq] using values

#axiomsClean kernel_native
#specNonCircular ordered avoiding [Bridge.grid,Execution.exec]
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked.RawCompatibility
