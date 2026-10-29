import bench.repairs.tritonbench_g.kldiv_ops.Misc.RealHost

namespace VeriTile.Bench.Repairs.KldivOps.RealExecution
noncomputable section

/-- Independent logical destination formula. The allocation tail is an input
because the source intentionally supplies no derivative value beyond target. -/
def destinationDerivative (flag : Bool) (count : Nat) (target allocation : Nat→ℝ) (i : Nat) : ℝ :=
  if i<count then backward flag (target i) else allocation i

/-- Independent complete returned tensor formula, preserving scalar-one object
return versus Torch's ordinary broadcast shape. Coordinates use the independently
proved row-major broadcast projection, not any source memory evaluator. -/
def returnedDerivative (flag scalarOne : Bool) (count : Nat) (outShape dstShape gradShape : List Nat)
    (target gradient allocation : Nat→ℝ) (i : Nat) : ℝ :=
  if scalarOne then destinationDerivative flag count target allocation i
  else destinationDerivative flag count target allocation (Host.broadcastIndex outShape dstShape i) *
    gradient (Host.broadcastIndex outShape gradShape i)

theorem destination_ideal (flag : Bool) (count : Nat) (target allocation : Nat→ℝ) (i : Nat) :
    Host.backwardKernelValue algebra flag count target allocation i=destinationDerivative flag count target allocation i := by
  unfold Host.backwardKernelValue destinationDerivative
  split
  · exact backward_scalar_ideal flag _
  · rfl

theorem returned_backward_ideal (flag scalarOne : Bool) (count : Nat) (outShape dstShape gradShape : List Nat)
    (target gradient allocation : Nat→ℝ) (i : Nat) :
    (if scalarOne then Host.backwardKernelValue algebra flag count target allocation i
    else libraryNumerics.multiply
      (Host.backwardKernelValue algebra flag count target allocation (Host.broadcastIndex outShape dstShape i))
      (gradient (Host.broadcastIndex outShape gradShape i)))=
      returnedDerivative flag scalarOne count outShape dstShape gradShape target gradient allocation i := by
  cases scalarOne <;> simp [returnedDerivative,libraryNumerics,destination_ideal]

#specNonCircular returnedDerivative avoiding [Host.backwardKernelValue,Backward.grid,Backward.gridModel,algebra]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.RealExecution
