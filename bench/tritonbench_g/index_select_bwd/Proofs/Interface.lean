import bench.tritonbench_g.index_select_bwd.IO
import bench.tritonbench_g.index_select_bwd.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Interface
open VeriTile.Triton FP.Structural
noncomputable section

def extend {A : Type} [Inhabited A] {B : Nat} (xs : Fin B → A) (i : Nat) : A :=
  if h : i < B then xs ⟨i, h⟩ else default

theorem implements {A : Type} [Inhabited A] (rest : Algebra A) (add : A → A → A)
    (p : Launch) (stride : p.stride = 1) (count : p.count ≤ p.block) :
    (programIO rest add p).Implements
      (fun values => result (fun j => add (current values j) (incoming values j))) := by
  apply ProgramIO.Implements.intro
  · intro values bounds hb s _
    change ScheduleSafe bounds [invocation rest add p] s
    rw [scheduleSafe_singleton]
    apply KernelProof.source_traceSafe
    · intro i active
      simpa only [stride, Nat.mul_one] using hb.2.1 ⟨0, by simp [programIO, memory, inputs, outputs]⟩ (i, PUnit.unit) active
    · intro i active
      exact hb.2.1 ⟨1, by simp [programIO, memory, inputs, outputs]⟩ (i, PUnit.unit) active
  · intro values s hin
    let initial := (invocation rest add p).initial s
    have inputs : KernelProof.InputValues p initial
        (extend (current values)) (extend (incoming values)) := by
      intro i active
      have inBlock : i < p.block := lt_of_lt_of_le active count
      constructor
      · simpa [initial, extend, inBlock, Invocation.initial_mem, mapValue]
          using hin ⟨0, by simp [programIO, memory, inputs, outputs]⟩ (⟨i, inBlock⟩, PUnit.unit) active
      · simpa [initial, extend, inBlock, Invocation.initial_mem, mapValue]
          using hin ⟨1, by simp [programIO, memory, inputs, outputs]⟩ (⟨i, inBlock⟩, PUnit.unit) active
    obtain ⟨final, ran, _, output, frame⟩ := KernelProof.successful rest add p initial
      (extend (current values)) (extend (incoming values)) stride count inputs
    refine ⟨final, ?_, ?_, ?_⟩
    · change runSchedule [invocation rest add p] s = some final
      rw [runSchedule_singleton]
      exact ran
    · intro o j active
      fin_cases o
      have value := output j.1.val active
      simpa [ProgramIO.OutputBinding, programIO, memory, outputs, result,
        mapCell, mapValue, sliceTarget, extend, j.1.isLt] using
        congrArg (mapCell (id : A → A)) value
    · intro r offset outside
      apply frame
      by_cases same : r = p.destination
      · right
        by_contra h
        have active : offset < p.count := by omega
        have inBlock : offset < p.block := lt_of_lt_of_le active count
        have miss := outside.1 ⟨0, by simp [programIO, memory, inputs, outputs]⟩ (⟨offset, inBlock⟩, PUnit.unit) active
        exact (miss.resolve_left (by simpa [outputs] using not_ne_iff.mpr same)) rfl
      · exact Or.inl same

#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Interface
