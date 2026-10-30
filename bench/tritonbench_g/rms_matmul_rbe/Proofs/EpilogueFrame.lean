import bench.tritonbench_g.rms_matmul_rbe.Proofs.Normalize
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Rotate
import bench.tritonbench_g.rms_matmul_rbe.Proofs.LoopFrame

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

theorem normalization_header {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Arguments) (s : State α) (epsilon : α)
    (moments accumulator : Values α .real [16,64])
    (he : s.regs .real [] "EPS"=some (fun _ => epsilon))
    (hm : s.regs .real [16,64] "x_sum"=some moments)
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    HeaderPreserved s (normalizationExecution a p s epsilon moments accumulator he hm ha).val := by
  intro d sh name hn
  simp [headerNames] at hn
  rcases hn with rfl|rfl|rfl|rfl|rfl
  all_goals simp [normalizationExecution]

theorem rotation_header {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Arguments) (s : State α) (theta : α) (pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    HeaderPreserved s (rotationExecution a p s theta pm pn accumulator ht hm hn ha).val := by
  intro d sh name hn
  simp [headerNames] at hn
  rcases hn with rfl|rfl|rfl|rfl|rfl
  all_goals simp [rotationExecution]

#axiomsClean normalization_header
#axiomsClean rotation_header
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
