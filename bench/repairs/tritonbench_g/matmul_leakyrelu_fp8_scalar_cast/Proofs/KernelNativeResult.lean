import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.KernelBuffers

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.KernelNativeResult
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
noncomputable section

/-- Complete launched kernel grid with its fixed-K8 native target and full raw-memory frame. -/
theorem kernel_equiv (M : Algebra Word) (native : PrimitiveOps) (instruction : InstructionProfile)
    (ports : LocalPorts native instruction) (law : Execution.ScalarLaws M native.operations)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (layout : KernelBuffers.Layout g b)
    (positiveK : g.K≠0) (scratch : State Word) (before : Memory Word) :
    ∃ final, Execution.grid M native.operations g.problem (b.views g) scratch before=some final ∧
      (∀ i j,i<g.M → j<g.N → final ((b.views g).C.address i j)=
        reference (fpReference instruction) (b.arguments g) before i j) ∧
      (∀ point,(∀ i<g.M,∀ j<g.N,(b.views g).C.address i j≠point) → final point=before point) := by
  refine ⟨Source.run (sourceArithmetic native.operations) g.problem (b.views g) before,
    Execution.grid_source M native.operations law g.problem (b.views g) layout scratch before,?_,?_⟩
  · intro i j hi hj
    have h := grid_reference native.operations (b.arguments g) (b.views g).C
      (by simpa only [KernelBuffers.arguments_problem] using layout) before positiveK i j hi hj
    simpa only [KernelBuffers.arguments_problem,operations_eq ports] using h
  · intro point outside
    exact Source.source_frame _ _ _ layout before point outside

theorem zero_fill (native : PrimitiveOps) (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers)
    (before : Memory Word) :
    ((zeroCode (b.views g).C).eval native.operations before).2=zeroMemory native.operations (b.views g).C before :=
  KernelBuffers.zero_fill native.operations g b before

#axiomsClean kernel_equiv
#axiomsClean zero_fill
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.KernelNativeResult
