import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelProfile

/-! Complete source execution and independent ordered native observations over
raw words and scalar ABI values. Reduction trees, FMA and operand order retain
the established compiler model without any additional numerical laws. -/
namespace VeriTile.Bench.Repairs.CrossEntropyOps.RawNative
open VeriTile.Triton FP.Structural KernelContract KernelProfile
noncomputable section

theorem native_forward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (scalars : RuntimeArgs.Values Word) (words : State Word)
    (positive : 0<p.block) (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (zLse : p.zloss≠p.lse) (zLoss : p.zloss≠p.loss) :
    let initial := entry p scalars words
    ∃ final, exec (KernelNative.engine (Compiler.algebra R)) (forwardSyntax p) initial=some final ∧
      Compiler.eval I S ((final.mem p.lse (p.lseBase+words.pids 1*p.rows+words.pids 0)).read .real)=
        Compiler.eval I S (forwardLSE R p scalars words) ∧
      Compiler.eval I S ((final.mem p.loss (p.lossBase+words.pids 1*p.rows+words.pids 0)).read .real)=
        Compiler.eval I S (forwardLoss R p scalars words) ∧
      (p.split=Bool.false →
        Compiler.eval I S ((final.mem p.zloss (p.zlossBase+words.pids 1*p.rows+words.pids 0)).read .real)=
          Compiler.eval I S (forwardZ R p scalars words)) ∧
      (∀ region offset,
        (region≠p.loss ∨ offset≠p.lossBase+words.pids 1*p.rows+words.pids 0) →
        (region≠p.lse ∨ offset≠p.lseBase+words.pids 1*p.rows+words.pids 0) →
        (p.split=Bool.true ∨ region≠p.zloss ∨ offset≠p.zlossBase+words.pids 1*p.rows+words.pids 0) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,zRead,frame⟩ := KernelNative.forward_outputs
    (Compiler.algebra R) p (entry p scalars words) positive separate outputs zLse zLoss
    (arguments scalars) (entry_binding p scalars words)
  refine ⟨final,execution,congrArg (Compiler.eval I S) lseRead,
    congrArg (Compiler.eval I S) lossRead,?_,frame⟩
  intro unsplit
  exact congrArg (Compiler.eval I S) (zRead unsplit)

theorem native_backward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (scalars : RuntimeArgs.Values Word) (words : State Word) :
    let initial := entry p scalars words
    ∃ final, exec (KernelNative.engine (Compiler.algebra R)) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, words.pids 1*p.block+lane.val<p.cols →
        Compiler.eval I S ((final.mem p.gradient
          (p.gradientBase+words.pids 0*p.gradientStride+(words.pids 1*p.block+lane.val))).read .real)=
            Compiler.eval I S (backwardLane R p scalars words lane)) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, words.pids 1*p.block+lane.val<p.cols →
          region≠p.gradient ∨ offset≠p.gradientBase+words.pids 0*p.gradientStride+(words.pids 1*p.block+lane.val)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,reads,frame⟩ := KernelNative.backward_outputs
    (Compiler.algebra R) p (entry p scalars words) (arguments scalars) (entry_binding p scalars words)
  refine ⟨final,execution,?_,frame⟩
  intro lane active
  exact congrArg (Compiler.eval I S) (reads lane active)

export KernelProfile (forward_square_fused backward_correction_fused)
#axiomsClean native_forward
#axiomsClean native_backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.RawNative
