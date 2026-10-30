import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelSafety

namespace VeriTile.Bench.Repairs.CrossEntropy2.KernelSafety
open VeriTile.Triton FP.Structural KernelNative
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward_tail (M : Algebra α) (p : Parameters) (initial : State α)
    (values : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial values)
    (bounds : RegionName → Nat)
    (gather : owns p initial →
      p.logitsBase+initial.pids 0*p.logitsStride+(target p initial-p.classStart).toNat<bounds p.logits)
    (loss : p.lossBase+initial.pids 1*p.rows+initial.pids 0<bounds p.loss)
    (zloss : p.zLossBase+initial.pids 1*p.rows+initial.pids 0<bounds p.zLoss) :
    Safety.traceSafeList (engine M) bounds (tailCode p) (ready M p initial) := by
  classical
  cases hs : p.hasSmoothing <;> cases sp : p.split
  all_goals by_cases ignored : target p initial=p.ignored
  all_goals by_cases owner : ownsTarget p initial
  all_goals by_cases hit : owns p initial
  all_goals
    have condition := hit
    simp only [owns,lt_min_iff] at condition
    simp [tailCode,forwardSyntax,hs,sp,ComputeKernel.surfaceBody,
      Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
      Safety.addressSafe,Safety.maskActive,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,numericEq,natLt,numericLt,numericLe,bop,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,Function.comp_def,
      abi.logitScale,abi.smoothing,abi.squareScale,abi.totalClasses,
      ready_lse,owner,ignored,condition,loss,zloss]
    all_goals first | exact lt_of_le_of_lt (Nat.sub_le _ _) (gather hit) | exact False.elim (owner (owns_local p initial hit))

theorem forward (M : Algebra α) (p : Parameters) (initial : State α)
    (positive : 0<p.block) (values : RuntimeArgs.Values α)
    (abi : RuntimeArgs.Bound p initial values) (bounds : RegionName → Nat)
    (label : p.labelsBase+initial.pids 0<bounds (Region.cast p.labels))
    (logits : ∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.cols →
      p.logitsBase+initial.pids 0*p.logitsStride+(initial.pids 1*p.block+lane.val)<bounds p.logits)
    (gather : owns p initial →
      p.logitsBase+initial.pids 0*p.logitsStride+(target p initial-p.classStart).toNat<bounds p.logits)
    (lse : p.lseBase+initial.pids 1*p.rows+initial.pids 0<bounds p.lse)
    (loss : p.lossBase+initial.pids 1*p.rows+initial.pids 0<bounds p.loss)
    (zloss : p.zLossBase+initial.pids 1*p.rows+initial.pids 0<bounds p.zLoss) :
    TraceSafe (engine M) bounds (forwardSyntax p) initial := by
  unfold TraceSafe
  rw [body_split,Safety.traceSafeList_append]
  refine ⟨forward_prefix M p initial positive values abi bounds label logits lse, ?_⟩
  intro after ran
  have eq := Option.some.inj (ran.symm.trans (forwardPrefix M p initial positive values abi))
  subst after
  exact forward_tail M p initial values abi bounds gather loss zloss

#axiomsClean forward_tail
#axiomsClean forward
end
end VeriTile.Bench.Repairs.CrossEntropy2.KernelSafety
