import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelBackward

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelExecution
open VeriTile.Triton FP.Structural KernelSource KernelModel
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

 theorem forward_outputs (M : Algebra α) (p : Parameters) (initial : State α)
    (positive : 0<p.block) (validLabel : label p initial = -100 ∨ 0≤label p initial)
    (distinct : p.loss≠p.lse) :
    ∃ final, exec (engine M) (forward p) initial=some final ∧
      (final.mem p.lse (initial.pids 0)).read .real=KernelModel.cast M (lseValue M p initial 0) ∧
      (final.mem p.loss (initial.pids 0)).read .real=KernelModel.cast M (forwardLoss M p initial) ∧
      (∀ region offset,
        (region≠p.loss ∨ offset≠initial.pids 0) →
        (region≠p.lse ∨ offset≠initial.pids 0) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,memory⟩ := forward_executes M p initial positive validLabel
  refine ⟨final,execution,?_,?_,?_⟩
  · rw [memory]
    simp [forwardMemory,State.write,Ne.symm distinct]
  · rw [memory]
    simp [forwardMemory,State.write]
  · intro region offset lossMiss lseMiss
    rw [memory]
    have lossMiss' : ¬(region=p.loss ∧ offset=initial.pids 0) := by tauto
    have lseMiss' : ¬(region=p.lse ∧ offset=initial.pids 0) := by tauto
    simp only [forwardMemory,State.write,lossMiss',lseMiss',if_false]

 theorem chunked_outputs (M : Algebra α) (p : Parameters) (initial : State α)
    (positive : 0<p.block) (validLabel : label p initial = -100 ∨ 0≤label p initial)
    (distinct : p.loss≠p.lse) :
    ∃ final, exec (engine M) (chunked p) initial=some final ∧
      (final.mem p.lse (initial.pids 0*p.chunks+initial.pids 1)).read .real=
        KernelModel.cast M (lseValue M p initial (initial.pids 1)) ∧
      (initial.pids 1=0 → (final.mem p.loss (initial.pids 0)).read .real=
        KernelModel.cast M (chunkLoss M p initial)) ∧
      (∀ region offset,
        (initial.pids 1=0 → region≠p.loss ∨ offset≠initial.pids 0) →
        (region≠p.lse ∨ offset≠initial.pids 0*p.chunks+initial.pids 1) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,execution,memory⟩ := chunked_executes M p initial positive validLabel
  refine ⟨final,execution,?_,?_,?_⟩
  · rw [memory]
    simp [chunkedMemory,State.write]
  · intro first
    rw [memory]
    simp [chunkedMemory,State.write,distinct,first]
  · intro region offset lossMiss lseMiss
    rw [memory]
    have lseMiss' : ¬(region=p.lse ∧ offset=initial.pids 0*p.chunks+initial.pids 1) := by tauto
    by_cases first : initial.pids 1=0
    · have lossMiss' : ¬(region=p.loss ∧ offset=initial.pids 0) := by
        have := lossMiss first
        tauto
      rw [first] at lseMiss'
      simp only [chunkedMemory,first,if_true,State.write,lseMiss',lossMiss',if_false]
    · simp only [chunkedMemory,first,if_false,State.write,lseMiss']

 theorem ignored_gradient (M : Algebra α) (p : Parameters) (initial : State α)
    (lane : Fin p.block) (ignored : label p initial = -100) :
    gradientValue M p initial lane=KernelModel.cast M (literal M 0) := by
  simp only [gradientValue,ignored,if_true]

#axiomsClean forward_outputs
#axiomsClean chunked_outputs
#axiomsClean ignored_gradient
#specNonCircular forwardLoss avoiding [forward,chunked,backward,FP.Structural.exec,FP.Structural.run]
#specNonCircular chunkLoss avoiding [forward,chunked,backward,FP.Structural.exec,FP.Structural.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelExecution
