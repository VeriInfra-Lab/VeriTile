import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Compiler

namespace VeriTile.Bench.Repairs.CrossEntropy2.RawNative
open VeriTile.Triton FP.Structural KernelContract KernelProfile
noncomputable section

/-! Successful complete source execution, native observation and the full frame
under raw memory/geometry/ABI premises. No Tensor views or admission arguments. -/
theorem forward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (initial : State Compiler.Term) (positive : 0<p.block)
    (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss ∧ p.lse≠p.zLoss ∧ p.loss≠p.zLoss)
    (values : RuntimeArgs.Values Compiler.Term) (abi : RuntimeArgs.Bound p initial values) :
    ∃ final, exec (KernelNative.engine (Compiler.algebra R)) (forwardSyntax p) initial=some final ∧
      Compiler.eval I S ((final.mem p.lse (p.lseBase+initial.pids 1*p.rows+initial.pids 0)).read .real)=
        Compiler.eval I S (KernelNative.lseValue (Compiler.algebra R) p initial) ∧
      Compiler.eval I S ((final.mem p.loss (p.lossBase+initial.pids 1*p.rows+initial.pids 0)).read .real)=
        Compiler.eval I S (KernelNative.lossValue (Compiler.algebra R) p initial) ∧
      Compiler.eval I S ((final.mem p.zLoss (p.zLossBase+initial.pids 1*p.rows+initial.pids 0)).read .real)=
        Compiler.eval I S (KernelNative.zValue (Compiler.algebra R) p initial) ∧
      (∀ region offset,
        (region≠p.loss ∨ offset≠p.lossBase+initial.pids 1*p.rows+initial.pids 0) →
        (region≠p.lse ∨ offset≠p.lseBase+initial.pids 1*p.rows+initial.pids 0) →
        (region≠p.zLoss ∨ offset≠p.zLossBase+initial.pids 1*p.rows+initial.pids 0) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,run,lse,loss,z,frame⟩ := KernelNative.forward_outputs (Compiler.algebra R) p initial positive separate outputs values abi
  exact ⟨final,run,congrArg (Compiler.eval I S) lse,congrArg (Compiler.eval I S) loss,
    congrArg (Compiler.eval I S) z,frame⟩

theorem backward (R : Compiler.Reductions) (I : Compiler.Instructions) (S : Compiler.Schedule)
    (p : Parameters) (initial : State Compiler.Term)
    (values : RuntimeArgs.Values Compiler.Term) (abi : RuntimeArgs.Bound p initial values) :
    ∃ final, exec (KernelNative.engine (Compiler.algebra R)) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.cols →
        Compiler.eval I S ((final.mem p.gradient (KernelNative.gradientOffset p initial (lane,PUnit.unit))).read .real)=
          Compiler.eval I S (KernelNative.gradientValue (Compiler.algebra R) p initial lane)) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.cols →
          region≠p.gradient ∨ offset≠KernelNative.gradientOffset p initial (lane,PUnit.unit)) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,run,values,frame⟩ := KernelNative.backward_outputs (Compiler.algebra R) p initial values abi
  exact ⟨final,run,fun lane active => congrArg (Compiler.eval I S) (values lane active),frame⟩

export KernelProfile (forward_square_fused backward_correction_fused)
#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.RawNative
