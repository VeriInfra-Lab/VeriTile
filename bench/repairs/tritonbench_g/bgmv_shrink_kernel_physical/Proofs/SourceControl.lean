import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.PairConnection

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.SourceControl
open VeriTile.Triton FP.Structural PairConnection
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 30000
variable {A : Type} [Inhabited A]

/-- The literal early return lowers to negated equality. Its active condition
has the same integer meaning as the earlier presentation's inequality. -/
theorem eval_not_eq (M : Algebra A) (p : Option ComputeDType)
    (a b : Op .int []) (s : State A) :
    evalOp M p (.boolNot (.eq .int .nil a b)) s =
      evalOp M p (.ne .int .nil a b) s := by
  have hn : evalOp M p (.boolNot (.eq .int .nil a b)) s =
      (evalOp M p (.eq .int .nil a b) s >>= fun value => some (fun i => !(value i))) := by
    rw [evalOp_unfold]
    rfl
  have he : evalOp M p (.eq .int .nil a b) s =
      (evalOp M p a s >>= fun va => evalOp M p b s >>= fun vb =>
        some (bop (fun (x y : Int) => decide (x=y)) .nil va vb)) := by
    rw [evalOp_unfold]
    rfl
  have hd : evalOp M p (.ne .int .nil a b) s =
      (evalOp M p a s >>= fun va => evalOp M p b s >>= fun vb =>
        some (bop (fun (x y : Int) => !(decide (x=y))) .nil va vb)) := by
    rw [evalOp_unfold]
    rfl
  rw [hn,he,hd]
  cases evalOp M p a s <;> cases evalOp M p b s <;> rfl

theorem early_return (M : Algebra A) (a b : Op .int [])
    {left right : List ComputeStmt} (body : CodeEq M left right) :
    StepEq M (.ifThen (.boolNot (.eq .int .nil a b)) left)
      (.ifThen (.ne .int .nil a b) right) := by
  constructor
  · intro s
    simp only [step,eval_not_eq]
    cases evalOp M none (.ne .int .nil a b) s with
    | none => rfl
    | some value =>
      simp only [bind,Option.bind_some]
      split
      · exact body.run s
      · rfl
  · intro bounds s
    have hn : Safety.safeOp M none bounds s (.boolNot (.eq .int .nil a b)) =
        Safety.safeOp M none bounds s (.eq .int .nil a b) := by
      rw [Safety.safeOp_unfold]
    have he : Safety.safeOp M none bounds s (.eq .int .nil a b) =
        (Safety.safeOp M none bounds s a ∧ Safety.safeOp M none bounds s b) := by
      rw [Safety.safeOp_unfold]
    have hd : Safety.safeOp M none bounds s (.ne .int .nil a b) =
        (Safety.safeOp M none bounds s a ∧ Safety.safeOp M none bounds s b) := by
      rw [Safety.safeOp_unfold]
    have safeCondition :
        Safety.safeOp M none bounds s (.boolNot (.eq .int .nil a b)) ↔
          Safety.safeOp M none bounds s (.ne .int .nil a b) := by rw [hn,he,hd]
    simp only [Safety.stepSafe_unfold,safeCondition,eval_not_eq]
    cases evalOp M none (.ne .int .nil a b) s with
    | none => rfl
    | some value => split <;> simp [body.safe]

/-- The copied source computes its constexpr stride as BLOCK_K * SPLIT_K.
The full signed loop, including failure and access safety, uses the exact
product 256 * 64; no loop iterations or accumulator operations are changed. -/
theorem constant_stride (M : Algebra A) (K : Nat) (body : List ComputeStmt) :
    StepEq M
      (.forRangeInt "k" (.constInt 0) (.constInt K)
        (.mul .nat .nil (.constNat 256) (.constNat 64)) body)
      (.forRangeInt "k" (.constInt 0) (.constInt K) (.constNat 16384) body) := by
  constructor
  · intro s
    simp [step,evalOp_unfold,numeric,bop]
  · intro bounds s
    simp [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,numeric,bop]

end
#axiomsClean eval_not_eq
#axiomsClean early_return
#axiomsClean constant_stride
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.SourceControl
