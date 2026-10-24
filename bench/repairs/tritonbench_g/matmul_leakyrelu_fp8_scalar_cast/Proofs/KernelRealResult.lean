import bench.repairs.tritonbench_g.matmul_leakyrelu_fp8_scalar_cast.Proofs.KernelBuffers

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.KernelRealResult
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
noncomputable section

/-- Complete typed-source grid, independent finite-K mathematics and full raw-memory frame. -/
theorem kernel_correct (M : Algebra ℝ) (law : Execution.ScalarLaws M realOps)
    (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (layout : KernelBuffers.Layout g b)
    (scratch : State ℝ) (before : Memory ℝ) :
    ∃ final, Execution.grid M realOps g.problem (b.views g) scratch before=some final ∧
      (∀ i j,i<g.M → j<g.N → final ((b.views g).C.address i j)=Real.formula g.problem (b.views g) before i j) ∧
      (∀ point,(∀ i<g.M,∀ j<g.N,(b.views g).C.address i j≠point) → final point=before point) := by
  refine ⟨Source.run (sourceArithmetic realOps) g.problem (b.views g) before,
    Execution.grid_source M realOps law g.problem (b.views g) layout scratch before,?_,?_⟩
  · intro i j hi hj
    rw [source_run_real]
    exact Real.output g.problem (b.views g) layout before i j hi hj
  · intro point outside
    exact Source.source_frame _ _ _ layout before point outside

theorem zero_fill (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) (before : Memory ℝ) :
    ((zeroCode (b.views g).C).eval realOps before).2=zeroMemory realOps (b.views g).C before :=
  KernelBuffers.zero_fill realOps g b before

#axiomsClean kernel_correct
#axiomsClean zero_fill
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.KernelRealResult
