import bench.tritonbench_g.fused_recurrent_delta.Proofs.RuntimeRegister
import bench.tritonbench_g.fused_recurrent_delta.RuntimeIO

namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeBridge
open VeriTile.Triton FP.Structural RuntimeTransport KernelProgram
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]

 theorem scaleAssignment (M : Algebra α) (scale : ℝ) (name : RegName) (hn : name ≠ "scale")
    (e : Op .real sh) (bc : Broadcast sh [] out) :
    Atom M (M.literal (some .fp32) .real scale)
      (.assign .real out name (.compute (.alg .fp32 (.mul .real bc e (.const scale)))))
      (.assign .real out name (.compute (.alg .fp32 (.mul .real bc e (.ref .real [] "scale"))))) := by
  apply assignment M _ name hn
  · intro s hs
    have same : FP.Structural.evalOp M (some .fp32) (.const scale) s=
        FP.Structural.evalOp M (some .fp32) (.ref .real [] "scale") s := hs.symm
    change FP.Structural.evalOp M (some .fp32) (.mul .real bc e (.const scale)) s=
      FP.Structural.evalOp M (some .fp32) (.mul .real bc e (.ref .real [] "scale")) s
    conv_lhs => rw [FP.Structural.evalOp]
    conv_rhs => rw [FP.Structural.evalOp]
    rw [same]
  · intro bounds s hs
    simp only [Safety.safeExpr,Safety.safeCompute]
    conv_lhs => rw [Safety.safeOp_unfold]
    conv_rhs => rw [Safety.safeOp_unfold]
    apply and_congr Iff.rfl
    conv_lhs => rw [Safety.safeOp_unfold]
    conv_rhs => rw [Safety.safeOp_unfold]


/-- The complete runtime source differs from the historical literal reference
only at the three actual scale reads. All loop/branch bodies are checked. -/
 theorem literal_pair (M : Algebra α) (kind : Kind) (s : Shape) (f : Flags) (scale : ℝ) :
    Pair (FP.ProfiledRewrite.engine M) (RuntimeIO.scaleValue M scale)
      (KernelProgram.source kind s f scale).surfaceBody (RuntimeIO.source kind Bool.false s f).surfaceBody := by
  cases kind <;> cases hb : f.vectorBeta
  all_goals
    simp only [KernelProgram.source,Source.forward,Source.backward,RuntimeIO.source,
      Runtime.forward,Runtime.backward,hb,Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody]
    repeat' first
      | apply samePair <;> simp [avoidsList,avoids]
      | exact RuntimeTransport.nil _ _
      | apply RuntimeTransport.cons
      | apply sameAssignment <;> decide
      | apply scaleAssignment <;> decide
      | apply sameStore
      | apply rangeAtom <;> (try decide)
      | apply dynamicAtom <;> (try decide)
      | apply ifAtom
      | apply barrier

end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.RuntimeBridge
