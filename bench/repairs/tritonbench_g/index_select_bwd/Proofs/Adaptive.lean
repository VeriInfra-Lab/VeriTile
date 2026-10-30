import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Interface
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Interface
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000

theorem tailBlock_covers (count : Nat) (small : count ≤ 512) : count ≤ tailBlock count := by
  unfold tailBlock
  split_ifs <;> omega

theorem tailBlock_bounded (count : Nat) : tailBlock count ≤ 512 := by
  unfold tailBlock
  split_ifs <;> omega

/-- The candidate executes the complete original JIT at a smaller block;
inactive lanes disappear while active reads/writes and the whole frame agree. -/
theorem adaptive_implements {A : Type} [Inhabited A] (rest : Algebra A) (add : A → A → A)
    (p : Launch) (stride : p.stride = 1) (count : p.count ≤ p.block) (small : p.count ≤ 512) :
    (adaptiveVariant rest add p).toProgramIO.Implements
      (fun values => result (fun j => add (current values j) (incoming values j))) := by
  have covered := tailBlock_covers p.count small
  apply ProgramIO.Implements.intro
  · intro values bounds hb s _
    change ScheduleSafe bounds [adaptiveInvocation rest add p] s
    rw [scheduleSafe_singleton]
    apply KernelProof.source_traceSafe
    · intro i active
      have inBlock : i.val < p.block := lt_of_lt_of_le active count
      simpa only [stride, Nat.mul_one] using hb.2.1
        ⟨0, by simp [adaptiveVariant, ProgramIO.Variant.toProgramIO, programIO, memory, inputs, outputs]⟩ (⟨i.val,inBlock⟩, PUnit.unit) active
    · intro i active
      have inBlock : i.val < p.block := lt_of_lt_of_le active count
      exact hb.2.1 ⟨1, by simp [adaptiveVariant, ProgramIO.Variant.toProgramIO, programIO, memory, inputs, outputs]⟩
        (⟨i.val,inBlock⟩, PUnit.unit) active
  · intro values s hin
    let initial := (adaptiveInvocation rest add p).initial s
    have inputs : KernelProof.InputValues {p with block := tailBlock p.count} initial
        (extend (current values)) (extend (incoming values)) := by
      intro i active
      have inBlock : i < p.block := lt_of_lt_of_le active count
      constructor
      · simpa [initial, extend, inBlock, Invocation.initial_mem, mapValue]
          using hin ⟨0, by simp [adaptiveVariant, ProgramIO.Variant.toProgramIO, programIO, memory, inputs, outputs]⟩ (⟨i,inBlock⟩, PUnit.unit) active
      · simpa [initial, extend, inBlock, Invocation.initial_mem, mapValue]
          using hin ⟨1, by simp [adaptiveVariant, ProgramIO.Variant.toProgramIO, programIO, memory, inputs, outputs]⟩ (⟨i,inBlock⟩, PUnit.unit) active
    obtain ⟨final, ran, _, output, frame⟩ := KernelProof.successful rest add
      {p with block := tailBlock p.count} initial
      (extend (current values)) (extend (incoming values)) stride covered inputs
    refine ⟨final, ?_, ?_, ?_⟩
    · change runSchedule [adaptiveInvocation rest add p] s = some final
      rw [runSchedule_singleton]
      exact ran
    · intro o j active
      fin_cases o
      have value := output j.1.val active
      simpa [ProgramIO.OutputBinding, adaptiveVariant, ProgramIO.Variant.toProgramIO, programIO, memory, outputs, result,
        mapCell, mapValue, sliceTarget, extend, j.1.isLt] using
        congrArg (mapCell (id : A → A)) value
    · intro r offset outside
      apply frame
      by_cases same : r = p.destination
      · right
        change p.count ≤ offset
        by_contra h
        have active : offset < p.count := by omega
        have inBlock : offset < p.block := lt_of_lt_of_le active count
        have miss := outside.1 ⟨0, by simp [adaptiveVariant, ProgramIO.Variant.toProgramIO, programIO, memory, outputs]⟩
          (⟨offset,inBlock⟩, PUnit.unit) active
        exact (miss.resolve_left (by simpa [outputs] using not_ne_iff.mpr same)) rfl
      · exact Or.inl same

theorem adaptive_equivalent {A : Type} [Inhabited A] (rest : Algebra A)
    (add : A → A → A) (p : Launch) (stride : p.stride = 1)
    (count : p.count ≤ p.block) (small : p.count ≤ 512) :
    (programIO rest add p).Equivalent (adaptiveVariant rest add p) :=
  ProgramIO.EquivalentOn.of_implementsOn
    (implements rest add p stride count) (adaptive_implements rest add p stride count small)

/-- Equality of every raw memory cell, allowing composition in the original
ascending index order without any assumption on the addition primitive. -/
theorem width_memory {A : Type} [Inhabited A] (rest : Algebra A) (add : A → A → A)
    (p : Launch) (s : State A) (stride : p.stride = 1)
    (count : p.count ≤ p.block) (small : p.count ≤ 512) :
    (Source.execution (KernelProof.arithmetic rest add) p s).val.mem =
      (Source.execution (KernelProof.arithmetic rest add)
        {p with block := tailBlock p.count} s).val.mem := by
  let current := fun i => (s.mem p.destination i).read .real
  let incoming := fun i => (s.mem p.incoming i).read .real
  obtain ⟨left, hl, ml, _⟩ := KernelProof.successful rest add p s current incoming
    stride count (fun _ _ => ⟨rfl,rfl⟩)
  obtain ⟨right, hr, mr, _⟩ := KernelProof.successful rest add
    {p with block := tailBlock p.count} s current incoming stride
    (tailBlock_covers p.count small) (fun _ _ => ⟨rfl,rfl⟩)
  have leftSame : (Source.execution (KernelProof.arithmetic rest add) p s).val = left :=
    Option.some.inj ((Source.execution (KernelProof.arithmetic rest add) p s).property.symm.trans hl)
  have rightSame : (Source.execution (KernelProof.arithmetic rest add)
      {p with block := tailBlock p.count} s).val = right :=
    Option.some.inj ((Source.execution (KernelProof.arithmetic rest add)
      {p with block := tailBlock p.count} s).property.symm.trans hr)
  rw [leftSame, rightSame, ml, mr]
  rfl

#axiomsClean adaptive_implements
#axiomsClean adaptive_equivalent
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Interface

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Program
open VeriTile.Triton FP.Structural Logical
noncomputable section

/-- The candidate changes only each actual slice's lane-block specialization.
The slice schedule, extraction/reinsertion map and input-row order are shared. -/
def adaptiveChunk {A : Type} [Inhabited A] {R N C : Nat} (rest : Algebra A)
    (add : A → A → A) (index : Fin N → Fin R) (h : Coord N C → A)
    (i : Fin N) (k : Nat) (g : Coord R C → A) : Coord R C → A :=
  let p := layout C k
  let result := (Source.execution (KernelProof.arithmetic rest add)
    {p with block := tailBlock p.count} (initial index h i k g)).val
  fun q => if q.1=index i ∧ q.2.val/512=k then
    (result.mem destinationRegion (q.2.val%512)).read .real else g q

theorem adaptiveChunk_eq {A : Type} [Inhabited A] {R N C : Nat} (rest : Algebra A)
    (add : A → A → A) (index : Fin N → Fin R) (h : Coord N C → A)
    (i : Fin N) (k : Nat) (g : Coord R C → A) :
    adaptiveChunk rest add index h i k g = chunk (KernelProof.arithmetic rest add) index h i k g := by
  have memory := Interface.width_memory rest add (layout C k) (initial index h i k g)
    rfl (Nat.min_le_left _ _) (Nat.min_le_left _ _)
  funext q
  unfold adaptiveChunk chunk
  rw [memory]

def adaptiveOneRow {A : Type} [Inhabited A] {R N C : Nat} (rest : Algebra A)
    (add : A → A → A) (index : Fin N → Fin R) (h : Coord N C → A)
    (i : Fin N) (g : Coord R C → A) : Coord R C → A :=
  (List.range ((C+511)/512)).foldl (fun state k => adaptiveChunk rest add index h i k state) g

def adaptiveRun {A : Type} [Inhabited A] {R N C : Nat} (rest : Algebra A)
    (add : A → A → A) (index : Fin N → Fin R) (h : Coord N C → A)
    (g : Coord R C → A) : Coord R C → A :=
  (List.finRange N).foldl (fun state i => adaptiveOneRow rest add index h i state) g

/-- All rows, every tail slice, duplicate indices and empty matrices retain
the original ordered arithmetic graph. -/
theorem adaptiveRun_eq {A : Type} [Inhabited A] {R N C : Nat} (rest : Algebra A)
    (add : A → A → A) (index : Fin N → Fin R) (h : Coord N C → A)
    (g : Coord R C → A) :
    adaptiveRun rest add index h g = run (KernelProof.arithmetic rest add) index h g := by
  simp only [adaptiveRun, adaptiveOneRow, adaptiveChunk_eq, run, oneRow]

#axiomsClean adaptiveRun_eq
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Program
