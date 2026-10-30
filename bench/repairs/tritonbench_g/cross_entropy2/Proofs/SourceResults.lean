import bench.repairs.tritonbench_g.cross_entropy2.Proofs.LogicalInputs

namespace VeriTile.Bench.Repairs.CrossEntropy2.SourceResults
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- Exact cell typing of the preserved forward execution, strengthening its
readback-only corollary without re-proving source execution. -/
theorem forward (M : Algebra α) (p : Parameters) (initial : State α)
    (positive : 0<p.block) (separate : p.lse≠p.logits)
    (outputs : p.lse≠p.loss ∧ p.lse≠p.zLoss ∧ p.loss≠p.zLoss)
    (args : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial args)
    (x : Nat → α) (label : Int)
    (binding : ∀ c, c<p.cols →
      (initial.mem p.logits (p.logitsBase+initial.pids 0*p.logitsStride+c)).read .real=x c)
    (labelBinding : KernelNative.target p initial=label) :
    ∃ final, exec (KernelNative.engine M) (forwardSyntax p) initial=some final ∧
      final.mem p.lse (p.lseBase+initial.pids 1*p.rows+initial.pids 0)=
        .mk .real (Reference.lse M p args x (initial.pids 1)) ∧
      final.mem p.loss (p.lossBase+initial.pids 1*p.rows+initial.pids 0)=
        .mk .real (Reference.loss M p args x (initial.pids 1) label) ∧
      final.mem p.zLoss (p.zLossBase+initial.pids 1*p.rows+initial.pids 0)=
        .mk .real (Reference.zLoss M p args x (initial.pids 1) label) ∧
      (∀ r o, (r≠p.loss ∨ o≠p.lossBase+initial.pids 1*p.rows+initial.pids 0) →
        (r≠p.lse ∨ o≠p.lseBase+initial.pids 1*p.rows+initial.pids 0) →
        (r≠p.zLoss ∨ o≠p.zLossBase+initial.pids 1*p.rows+initial.pids 0) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,memory⟩ := KernelNative.forward_executes M p initial positive separate args abi
  have logical := LogicalInputs.forward M p initial args abi x label binding labelBinding
  have lse := congrArg Prod.fst logical
  have loss := congrArg (fun z => z.2.1) logical
  have zloss := congrArg (fun z => z.2.2) logical
  refine ⟨final,ran,?_,?_,?_,?_⟩
  · rw [memory]
    simpa [KernelNative.outputMemory,State.write,outputs.1,outputs.2.1] using congrArg (Cell.mk .real) lse
  · rw [memory]
    simpa [KernelNative.outputMemory,State.write,outputs.2.2] using congrArg (Cell.mk .real) loss
  · rw [memory]
    simpa [KernelNative.outputMemory,State.write] using congrArg (Cell.mk .real) zloss
  · intro r o hm hl hz
    rw [memory]
    simp only [KernelNative.outputMemory,State.write]
    rw [if_neg (by tauto),if_neg (by tauto),if_neg (by tauto)]

theorem backward (M : Algebra α) (p : Parameters) (initial : State α)
    (args : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial args)
    (x : Nat → α) (label : Int) (L upstream : α)
    (binding : ∀ c, c<p.cols →
      (initial.mem p.logits (p.logitsBase+initial.pids 0*p.logitsStride+c)).read .real=x c)
    (labelBinding : KernelNative.target p initial=label)
    (lseBinding : (initial.mem p.lse (p.lseBase+initial.pids 0)).read .real=L)
    (upstreamBinding : label≠p.ignored →
      (initial.mem p.upstream (p.upstreamBase+initial.pids 0*p.upstreamStride)).read .real=upstream) :
    ∃ final, exec (KernelNative.engine M) (backwardSyntax p) initial=some final ∧
      (∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.cols →
        final.mem p.gradient (KernelNative.gradientOffset p initial (lane,PUnit.unit))=
          .mk .real (Reference.backward M p args (initial.pids 1) x label L upstream lane)) ∧
      (∀ r o, (∀ lane : Fin p.block, initial.pids 1*p.block+lane.val<p.cols →
        r≠p.gradient ∨ o≠KernelNative.gradientOffset p initial (lane,PUnit.unit)) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,memory⟩ := KernelNative.backward_executes M p initial args abi
  refine ⟨final,ran,?_,?_⟩
  · intro lane active
    rw [memory]
    have cell := State.masked_scatter_readback p.gradient (KernelNative.gradientOffset p initial)
      (fun i => Cell.mk .real (KernelNative.gradientValue M p initial i.1))
      (KernelNative.activeLane p initial) initial (lane,PUnit.unit) active
      (KernelNative.gradientOffset_injective p initial)
    change (KernelNative.gradientMemory M p initial).mem p.gradient _=_
    exact cell.trans (congrArg (Cell.mk .real)
      (LogicalInputs.backward M p initial args abi x label L upstream binding labelBinding lseBinding upstreamBinding lane))
  · intro r o outside
    rw [memory]
    apply State.masked_scatter_frame
    by_cases same : r=p.gradient
    · right
      intro lane _ active
      exact (outside lane.1 active).resolve_left (not_not_intro same)
    · exact Or.inl same

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.CrossEntropy2.SourceResults
