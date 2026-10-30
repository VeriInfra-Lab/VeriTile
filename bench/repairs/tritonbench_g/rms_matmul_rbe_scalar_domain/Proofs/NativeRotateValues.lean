import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.NativeRotate
import bench.repairs.tritonbench_g.rms_matmul_rbe_scalar_domain.Proofs.Reshape

namespace VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

theorem native_rotation_value {α : Type} [Inhabited α] (a : Arithmetic α)
    (plan : RotaryPlan) (p : Arguments) (s : State α) (theta : α) (pm pn : Nat)
    (accumulator : Values α .real [16,64])
    (ht : s.regs .real [] "THETA"=some (fun _ => theta))
    (hm : s.regs .nat [] "pid_m"=some (fun _ => pm))
    (hn : s.regs .nat [] "pid_n"=some (fun _ => pn))
    (ha : s.regs .real [16,64] "accumulator"=some accumulator) :
    (nativeRotationExecution a plan p s theta pm pn accumulator ht hm hn ha).val.regs
      .real [16,64] "accumulator"=some (nativeRotated a plan p theta pm pn accumulator) := by
  rcases plan with ⟨swap,fuse⟩
  cases swap <;> cases fuse
  all_goals
    simp [nativeRotationExecution,Arithmetic.model,binary,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
    funext i
    rcases i with ⟨row,lane,⟨⟩⟩
    simp only [reshape_join,reshape_split,kernelCast,ite_true]
    rw [reshape_join]
    dsimp only
    rw [reshape_split,reshape_split]
    simp [nativeRotated,nativeRotatedPair,orderedMul,orderedFma,
      pairAngle,pairColumn,evenLane,oddLane,
      reshape_join,reshape_split,pairOfLane,sideOfLane,pairLane,a.natLiteral]

#axiomsClean native_rotation_value
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
