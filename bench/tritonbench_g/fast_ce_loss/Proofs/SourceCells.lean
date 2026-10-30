import bench.tritonbench_g.fast_ce_loss.Proofs.KernelOutputs

namespace VeriTile.Bench.Repairs.FastCeLoss.SourceCells
open VeriTile.Triton FP.Structural KernelSource KernelModel KernelExecution
noncomputable section
variable {α : Type} [Inhabited α]

theorem forward (M : Algebra α) (p : Parameters) (s : State α)
    (positive : 0<p.block) (valid : label p s = -100 ∨ 0≤label p s)
    (distinct : p.loss≠p.lse) :
    ∃ final,exec (engine M) (KernelSource.forward p) s=some final ∧
      final.mem p.lse (s.pids 0)=.mk .real (KernelModel.cast M (lseValue M p s 0)) ∧
      final.mem p.loss (s.pids 0)=.mk .real (KernelModel.cast M (forwardLoss M p s)) ∧
      (∀ r o,(r≠p.loss ∨ o≠s.pids 0) → (r≠p.lse ∨ o≠s.pids 0) →
        final.mem r o=s.mem r o) := by
  obtain ⟨final,ran,memory⟩ := forward_executes M p s positive valid
  refine ⟨final,ran,?_,?_,?_⟩
  · rw [memory]
    simp [forwardMemory,State.write,Ne.symm distinct]
  · rw [memory]
    simp [forwardMemory,State.write]
  · intro r o hm hl
    rw [memory]
    simp only [forwardMemory,State.write]
    rw [if_neg (by tauto),if_neg (by tauto)]

theorem chunked (M : Algebra α) (p : Parameters) (s : State α)
    (positive : 0<p.block) (valid : label p s = -100 ∨ 0≤label p s)
    (distinct : p.loss≠p.lse) :
    ∃ final,exec (engine M) (KernelSource.chunked p) s=some final ∧
      final.mem p.lse (s.pids 0*p.chunks+s.pids 1)=
        .mk .real (KernelModel.cast M (lseValue M p s (s.pids 1))) ∧
      (s.pids 1=0 → final.mem p.loss (s.pids 0)=
        .mk .real (KernelModel.cast M (chunkLoss M p s))) ∧
      (∀ r o,(s.pids 1=0 → r≠p.loss ∨ o≠s.pids 0) →
        (r≠p.lse ∨ o≠s.pids 0*p.chunks+s.pids 1) → final.mem r o=s.mem r o) := by
  obtain ⟨final,ran,memory⟩ := chunked_executes M p s positive valid
  refine ⟨final,ran,?_,?_,?_⟩
  · rw [memory]
    simp [chunkedMemory,State.write]
  · intro first
    rw [memory]
    simp [chunkedMemory,State.write,distinct,first]
  · intro r o hm hl
    rw [memory]
    have missLSE : ¬(r=p.lse ∧ o=s.pids 0*p.chunks+s.pids 1) := by tauto
    by_cases first : s.pids 1=0
    · have missLoss : ¬(r=p.loss ∧ o=s.pids 0) := by have := hm first; tauto
      rw [first] at missLSE
      simp only [chunkedMemory,first,if_true,State.write,missLSE,missLoss,if_false]
    · simp only [chunkedMemory,first,if_false,State.write,missLSE]

theorem backward (M : Algebra α) (p : Parameters) (s : State α) :
    ∃ final,exec (engine M) (KernelSource.backward p) s=some final ∧
      (∀ lane : Fin p.block,s.pids 1*p.block+lane.val<p.vocab →
        final.mem p.logits (gradientOffset p s (lane,PUnit.unit))=
          .mk .real (gradientValue M p s lane)) ∧
      (∀ r o,(∀ lane : Fin p.block,s.pids 1*p.block+lane.val<p.vocab →
        r≠p.logits ∨ o≠gradientOffset p s (lane,PUnit.unit)) → final.mem r o=s.mem r o) := by
  obtain ⟨final,ran,memory⟩ := backward_executes M p s
  refine ⟨final,ran,?_,?_⟩
  · intro lane active
    rw [memory]
    exact State.masked_scatter_readback p.logits (gradientOffset p s)
      (fun i => Cell.mk .real (gradientValue M p s i.1)) (activeLane p s)
      s (lane,PUnit.unit) active (gradientOffset_injective p s)
  · intro r o outside
    rw [memory]
    apply State.masked_scatter_frame
    by_cases same : r=p.logits
    · right
      intro lane _ active
      exact (outside lane.1 active).resolve_left (not_not_intro same)
    · exact Or.inl same

#axiomsClean forward
#axiomsClean chunked
#axiomsClean backward
end
end VeriTile.Bench.Repairs.FastCeLoss.SourceCells
