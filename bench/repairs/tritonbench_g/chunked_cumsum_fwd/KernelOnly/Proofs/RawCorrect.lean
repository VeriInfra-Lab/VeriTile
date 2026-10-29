import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelObserved
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelRealTarget

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.RawReal
open VeriTile Triton FP.Structural
open Kernel Kernel.Scalar
noncomputable section

/-- Complete kernel execution, independent preparation/prefix formulas, and
raw memory frame. Geometry, scalar flags and initial reads are kernel inputs;
Tensor admission and physical readback are separate bridge results. -/
theorem kernel_correct
    (c : Geometry) (sizes : Sizes c) (plan : PrefixPlan c.BH c.BC)
    (a : InputWords ℝ) (s : State (Term ℝ)) (read : InputGraph s a)
    (hasBias reverseBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) :
    ∃ t, Kernel.launch (graphAlgebra (RealTarget.model reverseBias upper) c plan)
        c hasBias useSoftplus (.input lower) (.input (RealTarget.upperWord upper)) s = some t ∧
      (∀ b h k j, b < c.batches → h < c.heads → k < c.chunks → j < c.chunk →
        interpret (RealTarget.model reverseBias upper) ((t.mem "out" (packed c b h k j)).read .real) =
          RealTarget.prepared c (logicalInputs c a) hasBias useSoftplus lower upper b h k j ∧
        interpret (RealTarget.model reverseBias upper) ((t.mem "sum" (packed c b h k j)).read .real) =
          RealTarget.cumulative c (logicalInputs c a) hasBias useSoftplus lower upper b h k j) ∧
      Kernel.Frame c c.grid s t := by
  obtain ⟨t, ran, values, frame⟩ := Scalar.complete_from (RealTarget.model reverseBias upper)
    c sizes plan a s read hasBias useSoftplus lower (RealTarget.upperWord upper)
  refine ⟨t, ran, ?_, frame⟩
  intro b h k j hb hh hk hj
  simpa only [reference_prepared, reference_cumulative, RealTarget.preparation,
    RealTarget.cumulative_formula] using values b h k j hb hh hk hj

#axiomsClean kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.RawReal
