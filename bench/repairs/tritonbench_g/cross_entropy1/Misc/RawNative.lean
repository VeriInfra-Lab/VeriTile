import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelProfile

namespace VeriTile.Bench.Repairs.CrossEntropy1.RawNative
open VeriTile.Triton FP.Structural KernelContract KernelProfile
noncomputable section

/-- Both actual forward stores and the full raw typed-cell frame. Runtime
scalar words, reduction schedules, compiler fusion and primitive ports remain
explicit; there is no actual Tensor or Python admission parameter. -/
theorem forward_kernel (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (words : State Word) (scalars : RuntimeArgs.Values Word)
    (positive : 0<p.block) (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss) :
    let initial := entry p scalars words
    ∃ final, exec (KernelNative.engine (Compiler.algebra R)) (forwardSyntax p) initial=some final ∧
      Compiler.eval I S ((final.mem p.lse (p.lseBase+words.pids 1*p.rows+words.pids 0)).read .real)=
        Compiler.eval I S (forwardLSE R p scalars words) ∧
      Compiler.eval I S ((final.mem p.loss (p.lossBase+words.pids 1*p.rows+words.pids 0)).read .real)=
        Compiler.eval I S (forwardLoss R p scalars words) ∧
      (∀ region offset,
        (region≠p.loss ∨ offset≠p.lossBase+words.pids 1*p.rows+words.pids 0) →
        (region≠p.lse ∨ offset≠p.lseBase+words.pids 1*p.rows+words.pids 0) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,lseRead,lossRead,frame⟩ := KernelNative.forward_outputs
    (Compiler.algebra R) p (entry p scalars words) positive separate outputs
    (arguments scalars) (entry_binding p scalars words)
  exact ⟨final,execution,congrArg (Compiler.eval I S) lseRead,
    congrArg (Compiler.eval I S) lossRead,frame⟩

/-- Every native backward output and the complete raw memory frame, including
ignored labels, masked tails and unrestricted FP32 words. -/
theorem backward_kernel (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (words : State Word) (scalars : RuntimeArgs.Values Word) :
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
  exact ⟨final,execution,fun lane active => congrArg (Compiler.eval I S) (reads lane active),frame⟩

export KernelProfile (forward_square_fused backward_correction_fused)
#axiomsClean forward_kernel
#axiomsClean backward_kernel
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.RawNative
