import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.PairComputation
import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.PairIO

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.FPEquiv
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
open scoped VeriTile.Spec
noncomputable section
local instance : Inhabited Pair.Word := ⟨FP.Value.zero .fp32⟩

def original (M : Algebra Pair.Word) (ops : Operations Pair.Word) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (_layout : KernelBuffers.Layout g b) :=
  ProgramIO.Implementation.original (IO.matmulIO M ops g b) (fun _ => True)
def rightSlope (M : Algebra Pair.Word) (ops : Operations Pair.Word) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (_layout : KernelBuffers.Layout g b) :=
  ProgramIO.Implementation.ofVariant (PairIO.variant M ops g b) (fun _ => True)

/-- The selected complete source and its right-constant activation candidate
run the full grouped grid safely and successfully under the original IO bounds
and input binding, with equal typed output cells and both exact exterior frames.
FP32 multiplication uses the shared strict software certificate,
including NaNs, infinities and signed zeros. The unchanged ordered dot, casts
and other primitives remain explicit model parameters; no hardware-conformance
claim or experimental numerical rewrite is introduced. -/
specification source_pair_equiv (cfg : FP.Config) (rest : Algebra Pair.Word)
    (ops : Operations Pair.Word) (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers)
    (layout : KernelBuffers.Layout g b) :
    original (Pair.fp32Algebra cfg rest) ops g b layout ≡[[]]
      rightSlope (Pair.fp32Algebra cfg rest) ops g b layout := by
  exact ProgramIO.Implementation.of_equivalent (PairIO.variant _ ops g b)
    (PairIO.equivalent _ ops (Pair.fp32_mul_commutes cfg rest) g b layout)

#print_fp_assumptions source_pair_equiv
#axiomsClean source_pair_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.FPEquiv
