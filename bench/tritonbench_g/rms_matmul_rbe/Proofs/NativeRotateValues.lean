import bench.tritonbench_g.rms_matmul_rbe.Proofs.NativeRotate
import bench.tritonbench_g.rms_matmul_rbe.Proofs.Reshape

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
    (hm : s.regs .int [] "pid_m"=some (fun _ => (pm:Int)))
    (hn : s.regs .int [] "pid_n"=some (fun _ => (pn:Int)))
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
    rw [reshape_join row lane]
    dsimp only
    rw [reshape_split,reshape_split]
    simp [nativeRotated,nativeRotatedPair,orderedMul,orderedFma,
      pairAngle,pairColumn,evenLane,oddLane,
      reshape_join,reshape_split,pairOfLane,sideOfLane,pairLane,a.natLiteral]
    have nonnegative : 0≤(pn:Int)*64+2*((lane.val:Int)/2) := by positivity
    simp only [if_neg (not_lt_of_ge nonnegative),←Int.emod_def]

#axiomsClean native_rotation_value
end
end VeriTile.Bench.Repairs.RmsMatmulRbeScalarDomain
