import bench.repairs.tritonbench_g.log_softmax.Proofs.Kernel
import bench.tritonbench_g.log_softmax.Proofs.FP

namespace VeriTile.Bench.Repairs.LogSoftmax.Rewrite
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LogSoftmax
set_option maxRecDepth 20000

theorem forward_profiled [Inhabited A] (algebra : Algebra A) (O X : RegionName)
    (M N K BM C : Nat) (s : State A) :
    FP.Structural.exec (engine algebra) (Source.forward O X M N K BM C) s =
      FP.Structural.exec (engine algebra) (Source.forwardCommuted O X M N K BM C) s := by
  let original := (Source.forward O X M N K BM C).surfaceBody
  let candidate := (Source.forwardCommuted O X M N K BM C).surfaceBody
  have hprefix : run (engine algebra) (original.take 3) s =
      run (engine algebra) (candidate.take 3) s := by
    simp [original, candidate, Source.forward, Source.forwardCommuted,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,bop,State.setReg,ComputeDType.eraseDType,Int.mul_comm]
    funext dtype shape register
    split_ifs <;> try rfl
    all_goals subst_vars
    all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.mul_comm]
    all_goals funext i
    all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.mul_comm]
  have suffix : original.drop 3 = candidate.drop 3 := rfl
  change run (engine algebra) original s = run (engine algebra) candidate s
  conv_lhs => rw [← List.take_append_drop 3 original]
  conv_rhs => rw [← List.take_append_drop 3 candidate]
  rw [run_append,run_append,hprefix,suffix]

theorem backward_profiled [Inhabited A] (algebra : Algebra A) (Y G O : RegionName)
    (M N K BM C : Nat) (s : State A) :
    FP.Structural.exec (engine algebra) (Source.backward Y G O M N K BM C) s =
      FP.Structural.exec (engine algebra) (Source.backwardCommuted Y G O M N K BM C) s := by
  let original := (Source.backward Y G O M N K BM C).surfaceBody
  let candidate := (Source.backwardCommuted Y G O M N K BM C).surfaceBody
  have hprefix : run (engine algebra) (original.take 3) s =
      run (engine algebra) (candidate.take 3) s := by
    simp [original, candidate, Source.backward, Source.backwardCommuted,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,bop,State.setReg,ComputeDType.eraseDType,Int.mul_comm]
    funext dtype shape register
    split_ifs <;> try rfl
    all_goals subst_vars
    all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.mul_comm]
    all_goals funext i
    all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Int.mul_comm]
  have suffix : original.drop 3 = candidate.drop 3 := rfl
  change run (engine algebra) original s = run (engine algebra) candidate s
  conv_lhs => rw [← List.take_append_drop 3 original]
  conv_rhs => rw [← List.take_append_drop 3 candidate]
  rw [run_append,run_append,hprefix,suffix]

theorem engine_default (e : Kernel.Engine A C) : engine e.algebra=e.algebra := rfl

theorem forward_same_run [Inhabited A] (e : Kernel.Engine A C) (O X : RegionName)
    (M N K BM : Nat) (s : State A) :
    FP.Structural.exec e.algebra (Source.forward O X M N K BM C) s =
      FP.Structural.exec e.algebra (Source.forwardCommuted O X M N K BM C) s := by
  simpa only [engine_default] using forward_profiled e.algebra O X M N K BM C s

theorem backward_same_run [Inhabited A] (e : Kernel.Engine A C) (Y G O : RegionName)
    (M N K BM : Nat) (s : State A) :
    FP.Structural.exec e.algebra (Source.backward Y G O M N K BM C) s =
      FP.Structural.exec e.algebra (Source.backwardCommuted Y G O M N K BM C) s := by
  simpa only [engine_default] using backward_profiled e.algebra Y G O M N K BM C s

#axiomsClean forward_same_run
#axiomsClean backward_same_run
end VeriTile.Bench.Repairs.LogSoftmax.Rewrite
