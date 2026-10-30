import bench.tritonbench_g.softmax_flaggems.Proofs.FPReference

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPOneTile
open VeriTile Triton FP.Structural FPReference
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem backward_inner (M : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N B C : Nat) :
    ∃ final, exec M (Source.softmax_backward_kernel_inner_fp32 P G Y rows N C B Bool.true) s = some final ∧
      final.mem = (scatter s Y (backwardInnerOffset (s.pids 0) N)
        (backwardInnerActive (s.pids 0) rows N)
        (backwardInnerValue M s P G (s.pids 0) rows N B C)).mem := by
  simp [Source.softmax_backward_kernel_inner_fp32, exec, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, store, numeric, numericLt, numericLe,
    natLt, natLe, bop, ofFloat, toFloat, Region.cast, ComputeDType.eraseDType,
    State.setReg, Option.bind, backwardInnerValue, backwardInnerScale,
    backwardInnerLoad, backwardInnerOffset, backwardInnerActive,
    literal, FPReference.cast, mul, sub, FPReference.read, scatter]

  apply writeFold_mem_congr
  rfl

theorem backward_noninner (M : Algebra α) (s : State α) (P G Y : RegionName)
    (rows N K B C : Nat) :
    ∃ final, exec M (Source.softmax_backward_kernel_non_inner_fp32 P G Y rows N K B C Bool.true) s = some final ∧
      final.mem = (scatter s Y (noninnerOffset (s.pids 0) (s.pids 1) N K)
        (noninnerActive (s.pids 1) N K)
        (backwardNoninnerValue M s P G (s.pids 0) (s.pids 1) N K B C)).mem := by
  simp [Source.softmax_backward_kernel_non_inner_fp32, exec, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, store, numeric, numericLt, numericLe,
    natLt, natLe, bop, ofFloat, toFloat, Region.cast, ComputeDType.eraseDType,
    State.setReg, Option.bind, backwardNoninnerValue, backwardNoninnerScale,
    backwardNoninnerLoad, noninnerOffset, noninnerActive,
    literal, FPReference.cast, mul, sub, FPReference.read, scatter]
  apply writeFold_mem_congr
  rfl

theorem forward_inner (M : Algebra α) (s : State α) (X Y : RegionName)
    (rows N B : Nat) (hB : 0<B) :
    ∃ final, exec M (Source.forward_inner Y X rows N B Bool.true) s = some final ∧
      final.mem = (scatter s Y (innerOffset (s.pids 0) N) (innerActive N)
        (forwardInnerValue M s X (s.pids 0) N B)).mem := by
  simp [Source.forward_inner,
    VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FullSource.softmax_kernel_inner,
    exec, run, step, evalExpr, evalComputeOp, evalOp_unfold, store,
    numeric, numericLt, numericLe, natLt, natLe, bop, ofFloat, toFloat,
    Region.cast, ComputeDType.eraseDType, State.setReg, Option.bind,
    TileShape.axisDim, hB, forwardInnerValue, forwardInnerExp, forwardInnerLoad,
    innerOffset, innerActive, literal, FPReference.cast, mul, sub, FPReference.div,
    exp, FPReference.read, scatter]
  delta forwardInnerExp forwardInnerLoad forwardNoninnerExp forwardNoninnerLoad
  simp only [bop, Broadcast.leftIndex, Broadcast.rightIndex, innerActive, innerOffset, noninnerActive,
    noninnerOffset, FPReference.cast, FPReference.read, sub, exp, Bool.and_eq_true, decide_eq_true_eq]
  apply writeFold_mem_congr
  rfl

theorem forward_noninner (M : Algebra α) (s : State α) (X Y : RegionName)
    (rows N K B C : Nat) (hB : 0<B) :
    ∃ final, exec M (Source.forward_noninner Y X rows N K B C Bool.true) s = some final ∧
      final.mem = (scatter s Y (noninnerOffset (s.pids 0) (s.pids 1) N K)
        (noninnerActive (s.pids 1) N K)
        (forwardNoninnerValue M s X (s.pids 0) (s.pids 1) N K B C)).mem := by
  simp [Source.forward_noninner,
    VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FullSource.softmax_kernel_non_inner,
    exec, run, step, evalExpr, evalComputeOp, evalOp_unfold, store,
    numeric, numericLt, numericLe, natLt, natLe, bop, ofFloat, toFloat,
    Region.cast, ComputeDType.eraseDType, State.setReg, Option.bind,
    TileShape.axisDim, hB, forwardNoninnerValue, forwardNoninnerExp, forwardNoninnerLoad,
    noninnerOffset, noninnerActive, literal, FPReference.cast, mul, sub, FPReference.div,
    exp, FPReference.read, scatter]
  delta forwardInnerExp forwardInnerLoad forwardNoninnerExp forwardNoninnerLoad
  simp only [bop, Broadcast.leftIndex, Broadcast.rightIndex, innerActive, innerOffset, noninnerActive,
    noninnerOffset, FPReference.cast, FPReference.read, sub, exp, Bool.and_eq_true, decide_eq_true_eq]
  apply writeFold_mem_congr
  rfl

#axiomsClean backward_inner
#axiomsClean backward_noninner
#axiomsClean forward_inner
#axiomsClean forward_noninner

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPOneTile
