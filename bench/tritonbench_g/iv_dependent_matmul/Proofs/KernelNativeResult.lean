import bench.tritonbench_g.iv_dependent_matmul.Proofs.Interface

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.KernelNativeResult
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def backend (rest : Algebra Word) (profile : Target.NativeProfile) : Algebra Word :=
  FPValue.backend rest .ieeeValue
    (Instructions.dotPort (Instructions.referenceInstructions profile.reference))
    (Instructions.accumulatorPort (Instructions.referenceInstructions profile.reference))

/-- Complete source grid and independent TF32 K8/carry/conversion target on
raw words, with every exterior typed-memory cell preserved. -/
theorem kernel_native (rest : Algebra Word) (profile : Target.NativeProfile)
    (g : Kernel.Geometry) (initial : State Word) (a : Allocation g.request initial)
    (d : Index64.Addressable g.request initial a) (l : Index64.Launchable g.request) :
    ∃ final,Kernel.grid (backend rest profile) .accumulator g.request initial=some final ∧
      (∀ i,i<g.rows → ∀ j,j<g.columns → final.mem g.C (offset g.request i j)=
        Cell.mk .fp32 (Target.native profile.reference g.inner g.blockInner
          (Target.inputA g.request initial) (Target.inputB g.request initial) i j)) ∧
      (∀ region o,¬Footprint g.request region o → final.mem region o=initial.mem region o) := by
  let R:=profile.reference
  let D:=Instructions.dotPort (Instructions.referenceInstructions R)
  let F:=Instructions.accumulatorPort (Instructions.referenceInstructions R)
  obtain ⟨final,run,memory⟩:=Kernel.native_grid rest R.cfg D F g.request initial a d l initial
  obtain ⟨_,_,_,values,frame⟩:=Execution.host_projection rest R.cfg D F .accumulator g.request initial a d l
  refine ⟨final,run,?_,?_⟩
  · intro i hi j hj
    rw [memory]
    have value:=values i hi j hj
    change _=Cell.mk .fp32 (reference R.cfg D F .accumulator g.request initial i j) at value
    rw [Target.reference_eq R g.request initial i j hi hj] at value
    exact value
  · intro region o outside
    rw [memory]
    exact frame region o outside

#axiomsClean kernel_native
#specNonCircular Target.native avoiding [Kernel.grid,Execution.program]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.KernelNativeResult
